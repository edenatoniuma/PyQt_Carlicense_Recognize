#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :process_data.py
# @Time      :2024/5/14 15:13
# @Author    :嘉隆

import os
import random
import shutil


class SplitData:
    def __init__(self, img_file, labels_file):
        self.img_file = img_file
        self.labels_file = labels_file
        self.img_ls = os.listdir(self.img_file)
        self.labels_ls = os.listdir(self.labels_file)
        self.img_no_name = [os.path.splitext(filename)[0] for filename in self.img_ls]
        self.labels_no_name = [os.path.splitext(filename)[0] for filename in self.labels_ls]
        self.intersection = None

    def remove_different_elements(self):
        self.intersection = list(set(self.img_no_name) & set(self.labels_no_name))

    def split_img(self, test_size):
        random.seed(42)
        random.shuffle(self.intersection)
        train = self.intersection[:int(len(self.intersection) * (1 - test_size))]
        val = test = self.intersection[:int(len(self.intersection) * test_size)]
        with open("../datasets/dataset/train.txt", "a") as train_txt, open("../datasets/dataset/val.txt",
                                                                           "a") as val_txt, open(
            "../datasets/dataset/test.txt", "a") as test_txt:
            for train_img in train:
                train_txt.write(f"F:/AICAR/total/{train_img}.jpg\n")
            for val_img in val:
                val_txt.write(f"F:/AICAR/total/{val_img}.jpg\n")
                test_txt.write(f"F:/AICAR/total/{val_img}.jpg\n")


if __name__ == "__main__":
    split_data = SplitData(img_file="./total", labels_file="../datasets/labels")
    split_data.remove_different_elements()
    split_data.split_img(test_size=0.2)
