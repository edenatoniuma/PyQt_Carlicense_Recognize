#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :car_license_enhance.py
# @Time      :2024/5/20 10:10
# @Author    :嘉隆
import random

import cv2
import numpy as np


def bgr2gray(img):
    image = cv2.imread(img)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.bilateralFilter(gray_img, d=3, sigmaColor=1, sigmaSpace=5)
    edges = cv2.Canny(filtered_image, threshold1=40, threshold2=90)
    ret, car_binary = cv2.threshold(edges.copy(), 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(car_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((car_binary.shape[0], car_binary.shape[1], 3), dtype=np.uint8)
    # sort_contours = ([contour for contour in contours if len(contour) > 220])
    test_contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)
    current_contours = test_contours[:1]
    print(len(current_contours[0][:]))
    for i in range(len(contours)):
        if len(contours[i]) == len(current_contours[0][:]):
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))

            cv2.drawContours(drawing, contours, i, color, 1)
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.imshow('rectangle', image)
    cv2.imshow('contours', drawing)
    cv2.imshow('binary', car_binary)
    cv2.imwrite('images/binary_img.jpg', car_binary)
    cv2.imwrite("images/filtered_img.jpg", filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imshow("origin_gray_image", gray_img)
    # cv2.imshow("bilateral_filtered_image", filtered_image)

    # cv2.imwrite("images/filtered_img.jpg", filtered_image)
    # cv2.imwrite("images/origin_gray_img.jpg", gray_img)
    # cv2.imwrite("images/edges.jpg", edges)
    # print(max(filtered_image))


if __name__ == "__main__":
    bgr2gray("images/resize_img.jpg")
