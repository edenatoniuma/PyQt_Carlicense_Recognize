#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :conver_size.py
# @Time      :2024/6/3 11:16
# @Author    :嘉隆
import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    resize_img = cv2.resize(img, (94, 24))
    cv2.imshow('94*24', resize_img)
    cv2.imwrite('./resize_img.jpg', resize_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
