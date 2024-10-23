#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :call_detect.py
# @Time      :2024/6/19 11:08
# @Author    :嘉隆
import sys
import os
import cv2
from ultralytics import YOLO
from detect_model.my_LPRNet import ReadLicensePlate
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from login_register.detect import Ui_Form
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal
import warnings

warnings.filterwarnings("ignore")


class DetectWindow(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(DetectWindow, self).__init__(parent)
        # UI初始化
        self.setupUi(self)
        # 按钮初始化
        self.select_img_btn.clicked.connect(self.upload_file)
        # 绑定多线程触发事件
        self.detect_btn.clicked.connect(self.start_detect)
        # 定义文件路径变量
        self.filename = ""
        # 窗口显示
        self.show()
        # 初始化线程
        self.thread = None

    # 上传文件
    def upload_file(self):
        options = QFileDialog.Options()
        self.filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                       "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif)", options=options)
        if self.filename:
            pixmap = QPixmap(self.filename)
            self.original_img_label.setPixmap(pixmap.scaled(self.original_img_label.size()))

    def start_detect(self):
        # 创建线程
        self.thread = Worker(img_path=self.filename)
        self.thread._signal.connect(self.call_backlog)
        self.thread.start()

    # 接收信号
    @pyqtSlot(list)
    def call_backlog(self, value):
        if value:
            self.detect_object_text.setText(str(value[1]))
            self.read_license_plate_text.setText(value[0])
            detect_pixmap = QPixmap("G:/qt_designer/detect_model/imgs/detect/result.jpg")
            license_plate_pixmap = QPixmap("G:/qt_designer/detect_model/imgs/license_plate/1.jpg")
            self.predict_result_label.setPixmap(detect_pixmap.scaled(self.predict_result_label.size()))
            self.read_license_plate_label.setPixmap(license_plate_pixmap.scaled(self.read_license_plate_label.size()))


class Worker(QThread):
    # 定义发射信号
    _signal = pyqtSignal(list)

    def __init__(self, weights='G:/qt_designer/yolov8/runs/detect/train4/weights/best.pt',
                 yaml=None, img_path=None):
        super(Worker, self).__init__()
        self.img_path = img_path
        self.weights = weights
        self.yaml = yaml
        self.idx2cls_name = {
            0: "car",
            1: "license_plate",
            2: "fire_extinguisher",
            3: "tripod"
        }

    def predict(self, img_path):
        self.model = YOLO(self.weights)
        self.result = self.model(img_path)
        return self.result

    def save_img(self, save_dir, predictions, id_ls, image_path, cls_index):
        """
        :param cls_index:
        :param save_dir:
        :param predictions:
        :param id_ls:
        :param image_path:
        :return:
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        origin_img = cv2.imread(image_path,
                                cv2.COLOR_BGR2RGB)
        for _ in id_ls:
            project_bbox = predictions.data.tolist()[_]
            pos1, pos2 = (int(project_bbox[0]), int(project_bbox[1])), (int(project_bbox[2]), int(project_bbox[3]))
            # 随机颜色
            # color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            cv2.rectangle(origin_img, pos1, pos2, (0, 255, 0), 2)
            cls_idx = cls_index[_]
            cls_name = self.idx2cls_name[cls_idx]
            cv2.putText(origin_img, cls_name, (pos1[0] - 10, pos1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                        2)
        cv2.imwrite(os.path.join(save_dir, "result.jpg"), origin_img)

    def __del__(self):
        self.wait()

    def run(self):
        result_dict = dict()
        """
        接受一个图像路径，返回检测结果和车牌裁剪图像
        :param img_path:
        :return:
        """
        results = self.predict(self.img_path)
        for result in results:
            img_path = result.path
            bbox = result.boxes
            try:
                cls_list = bbox.cls.tolist()
                idx_ls = list()
                # 列表保存每个类别置信度最高的下标
                for i in range(4):
                    idx_ls.append(cls_list.index(i))
                license_plate_idx = cls_list.index(1)
            except ValueError as E:
                print("未找到指定类别bbox")
                continue
            #  保存检测结果
            self.save_img("G:/qt_designer/detect_model/imgs/detect", bbox, idx_ls, img_path, cls_list)
            license_plate_bbox = bbox.data.tolist()[license_plate_idx]
            image = cv2.imread(img_path,
                               cv2.COLOR_BGR2RGB)
            p1, p2 = (int(license_plate_bbox[0]), int(license_plate_bbox[1])), (int(license_plate_bbox[2]),
                                                                                int(license_plate_bbox[3]))
            car_license_img = image[p1[1]: p2[1], p1[0]: p2[0]]
            resize_license_plate = cv2.resize(car_license_img, (94, 24))

            # 写入车牌图像
            cv2.imwrite("G:/qt_designer/detect_model/imgs/license_plate/1.jpg", resize_license_plate)
            # 调用识别车牌类
            predict = ReadLicensePlate(yaml_file="G:/qt_designer/detect_model/config.yaml")
            result_str = predict.test()
            for _ in cls_list:
                cls_name = self.idx2cls_name.get(_, "unknown")
                result_dict[cls_name] = result_dict.get(cls_name, 0) + 1
            result_dict["speed"] = result.speed
            self._signal.emit([result_str, result_dict])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_form = DetectWindow()
    sys.exit(app.exec_())
