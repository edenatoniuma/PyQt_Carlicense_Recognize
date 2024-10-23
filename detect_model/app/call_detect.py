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
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QVBoxLayout, QWidget, QStatusBar
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtGui import QPixmap, QMovie
from login_register.detect import Ui_Form
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal
import warnings

warnings.filterwarnings("ignore")


class DetectWindow(QWidget, Ui_Form):
    def __init__(self, parent=None, is_admin=None):
        super(DetectWindow, self).__init__(parent)
        # UI初始化
        self.choice_window = None
        self.setupUi(self)
        self.is_admin = is_admin
        # 按钮初始化
        self.select_img_btn.clicked.connect(self.upload_file)
        self.back_btn.clicked.connect(self.back_choice_window)
        # 绑定多线程触发事件，且文件名为空的情况下按钮不可使用
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.start_detect)
        # 定义文件路径变量
        self.filename = ""
        # 添加加载动画
        self.movie = QMovie("./img/Rhombus.gif")
        self.load_gif_label.setMovie(self.movie)
        # 窗口显示
        self.center()
        self.show()
        # 初始化线程
        self.thread = None

    # 上传文件
    def upload_file(self):
        options = QFileDialog.Options()
        self.filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                       "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif)", options=options)
        # 路径存在
        if self.filename:
            pixmap = QPixmap(self.filename)
            self.original_img_label.setPixmap(pixmap.scaled(self.original_img_label.size()))
            # 检测按钮激活
            self.detect_btn.setEnabled(True)
        # 路径不存在则弹窗警告
        else:
            QMessageBox.warning(self, "Warning", "File Can't Be None!")

    def start_detect(self):
        # 创建线程
        self.movie = QMovie("./img/Rhombus.gif")
        self.load_gif_label.setMovie(self.movie)
        self.thread = Worker(img_path=self.filename)
        # 信号量连接call_backlog槽函数，用于接受发射的信号并输出
        self.thread._signal.connect(self.call_backlog)
        self.thread.start()

        self.movie.start()
        self.detect_btn.setEnabled(False)

    # 接收信号
    @pyqtSlot(list)
    def call_backlog(self, value):
        if value:
            self.detect_object_text.setText(str(value[1]))
            self.read_license_plate_text.setText(value[0])
            # detect_pixmap = QPixmap("G:/qt_designer/detect_model/imgs/detect/result.jpg")
            # license_plate_pixmap = QPixmap("G:/qt_designer/detect_model/imgs/license_plate/1.jpg")
            # self.predict_result_label.setPixmap(detect_pixmap.scaled(self.predict_result_label.size()))
            # self.read_license_plate_label.setPixmap(license_plate_pixmap.scaled(self.read_license_plate_label.size()))
            self.movie.stop()
            self.load_gif_label.clear()
            self.detect_btn.setEnabled(True)
            
    def back_choice_window(self):
        from call_choice_fun import ChoiceFunWindow
        self.choice_window = ChoiceFunWindow(is_admin=self.is_admin)
        self.hide()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '确认关闭', '你确定要关闭程序吗?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)


# 子线程
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

    # 重写Qthread的Run函数
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
            detect_result_str = (
                f"耗时：\n{result_dict['speed']}\n"
                f"年检结果如下：\n车牌：{result_dict['license_plate']}\n"
                f"灭火器：{result_dict['fire_extinguisher']}\n三脚架：{result_dict['tripod']}\n"
                f"是否通过：{all(value for value in result_dict.values())}")
            # 检测完成，发射信号
            self._signal.emit([result_str, detect_result_str])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_form = DetectWindow()
    sys.exit(app.exec_())
