import os
import shutil
from tqdm import tqdm


def move_images(source_folder, destinaton_folder):
    if not os.path.exists(destinaton_folder):
        os.makedirs(destinaton_folder)

    for file in tqdm(os.listdir(source_folder), desc="load_file"):
        source_file = os.path.join(source_folder, file)
        for image in os.listdir(source_file):
            source_img = os.path.join(source_file, image)
            destinaton_img = os.path.join(destinaton_folder, image)
            shutil.move(source_img, destinaton_img)


if __name__ == "__main__":
    move_images(source_folder="C:\\Users\\付超俊\\PycharmProjects\\yolov8\\generateCarPlate-master\\gen_res", destinaton_folder="E:\\资源\\汽车识别模型\\car_license")
