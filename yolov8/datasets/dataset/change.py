#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :change.py
# @Time      :2024/5/15 14:33
# @Author    :嘉隆
import os

if __name__ == "__main__":
    with open('test.txt', 'r') as f, open('F:/AICAR/dataset/test.txt', 'a') as f2:
        data = f.readlines()
        for img in data:
            info = img.split('/')[-1]
            f2.write('F:/AICAR/dataset/images/'+info)
