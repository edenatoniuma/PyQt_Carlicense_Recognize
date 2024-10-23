import os.path

from ultralytics import YOLO
import cv2


class Detect:
    def __init__(self, weights, yaml=None, epochs=None, data=None, img=None, resume=False):
        self.weights = weights
        self.yaml = yaml
        self.epochs = epochs
        self.data = data
        self.img = img
        self.model = YOLO(weights)
        self.resume = resume
        self.metrics = None
        self.result = None

    def train(self):
        self.model.train(data=self.yaml, epochs=self.epochs, resume=self.resume)
        self.metrics = self.model.val()

    def predict(self):
        self.result = self.model(self.img, stream=True)
        return self.result


if __name__ == "__main__":
    # model = YOLO('runs/detect/train4/weights/best.pt')
    # img_path = "images/17091452001500590325_1_guiCCHN360.jpg"
    img_dir = "test_fps/*.jpg"
    # img_path = "C:/Users/付超俊/Desktop/0794850.jpg"
    # detect = Detect(weights='yolov8n.pt', yaml="F:/AICAR/dataset/train.yaml", epoch=40)
    # detect.train()
    detect = Detect(weights='runs/detect/train4/weights/best.pt', img=img_dir)
    results = detect.predict()
    ms = 0
    for result in results:
        ms += result.speed['inference']
    print(ms/60)
    #     img_path = result.path
    #     bbox = result.boxes
    #     try:
    #         license_plate_idx = bbox.cls.tolist().index(1)
    #     except ValueError as E:
    #         print("未找到指定类别bbox，跳到下一张图。")
    #         continue
    #     license_plate_bbox = bbox.data.tolist()[license_plate_idx]
    #     image = cv2.imread(img_path,
    #                        cv2.COLOR_BGR2RGB)
    #     p1, p2 = (int(license_plate_bbox[0]), int(license_plate_bbox[1])), (int(license_plate_bbox[2]),
    #                                                                         int(license_plate_bbox[3]))
    #     car_license_img = image[p1[1]: p2[1], p1[0]: p2[0]]
    #     resize_license_plate = cv2.resize(car_license_img, (94, 24))
    #     file_name = os.path.splitext(img_path)[0].split("\\")[-1].split("_")[-1] + ".jpg"
    #     cv2.imwrite("F:/AICAR/dataset/license_plate/"+file_name, resize_license_plate)
        # cv2.imshow("car", car_license_img)
    # for res in result[0].boxes.data:
    #     if res[-1] == 1:
    #         p1, p2 = (int(res[0]), int(res[1])), (int(res[2]), int(res[3]))
    #         draw = cv2.rectangle(image, p1, p2, (255, 0, 0), 2)
    #         car_license_img = image[p1[1]:p2[1], p1[0]:p2[0]]
    #         # resize_img = cv2.resize(car_license_img, (94, 24), cv2.INTER_AREA)
    #         cv2.imshow("Image with a Car License", car_license_img)
    #         cv2.imwrite('C:/Users/付超俊/PycharmProjects/yolov8/images/test.jpg', car_license_img)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
