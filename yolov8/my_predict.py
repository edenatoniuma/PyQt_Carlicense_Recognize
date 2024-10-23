import os
import random
import re

from ultralytics import YOLO
import cv2


class Detect:
    def __init__(self, weights, yaml=None, img=None):
        self.weights = weights
        self.yaml = yaml
        self.img = img
        self.model = YOLO(weights)
        self.result = None

    def predict(self):
        self.result = self.model(self.img)
        return self.result


def save_img(save_dir, predictions, id_ls, image_path, idx2cls_name, cls_index):
    """
    :param cls_index:
    :param idx2cls_name:
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
        cls_name = idx2cls_name[cls_idx]
        cv2.putText(origin_img, cls_name, (pos1[0] - 10, pos1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(save_dir, "result.jpg"), origin_img)


def rename(img_path):
    chars = {
        "gui": "贵",
        "gan": "赣",
        "chuan": "川",
        "e": "鄂",
        "hu": "沪",
        "ji": "冀",
        "jing": "京",
        "jin": "晋",
        "liao": "辽",
        "lu": "鲁",
        "meng": "蒙",
        "min": "闽",
        "shan": "陕",
        "su": "苏",
        "wan": "皖",
        "xiang": "湘",
        "yu": "渝",
        "yue": "粤",
        "yun": "云",
        "zhe": "浙",
        "zang": "藏",
    }
    pattern = re.compile(r"[a-z]")
    file_name_with_extension = os.path.basename(img_path)
    file_name = os.path.splitext(file_name_with_extension)[0]
    lower_char = "".join(pattern.findall(file_name))
    try:
        char2cn = chars[lower_char]
        new_file_name = re.sub(r"[a-z][A-Z]?", "", file_name)
        char2cn_name = char2cn + new_file_name
        return char2cn_name
        # source_path = img_path
        # des_path = os.path.join(os.path.dirname(img_path), char2cn + new_file_name + ".jpg")
        # if os.path.exists(des_path):
        #     print("file is exist")
        # os.rename(source_path, des_path)
    except KeyError as e:
        print(f"Error: {e}")


def main(img_path="./test_images/18010252001500682656_1_guiCCVD862.jpg"):
    """
    接受一个图像路径，返回检测结果和车牌裁剪图像
    :param img_path:
    :return:
    """
    idx2cls_dict = {
        0: "car",
        1: "license_plate",
        2: "fire_extinguisher",
        3: "tripod"
    }
    weights_path = 'runs/detect/train4/weights/best.pt'

    detect = Detect(weights=weights_path, img=img_path)
    results = detect.predict()
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
        save_img("./test_images", bbox, idx_ls, img_path, idx2cls_dict, cls_list)
        license_plate_bbox = bbox.data.tolist()[license_plate_idx]
        image = cv2.imread(img_path,
                           cv2.COLOR_BGR2RGB)
        p1, p2 = (int(license_plate_bbox[0]), int(license_plate_bbox[1])), (int(license_plate_bbox[2]),
                                                                            int(license_plate_bbox[3]))
        car_license_img = image[p1[1]: p2[1], p1[0]: p2[0]]
        resize_license_plate = cv2.resize(car_license_img, (94, 24))
        # 只是预测不用修改文件名
        # origin_file_name = os.path.splitext(img_path)[0].split("\\")[-1].split("_")[-1] + ".jpg"
        # file_name = rename(origin_file_name)
        cv2.imwrite("./test_images/result_license_plate.jpg", resize_license_plate)


if __name__ == "__main__":
    main()
    # rename("./test_images/guiCCVD862.jpg")
