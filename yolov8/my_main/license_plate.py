#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :license_plate.py
# @Time      :2024/6/12 9:37
# @Author    :嘉隆

# -*- coding: utf-8 -*-
# /usr/bin/env/python3

"""
test pretrained model.
Author: aiboy.wei@outlook.com .
"""

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os
from torchvision import transforms


class ReadLicense:
    def __init__(self, img_path, pretrained_model, cuda=True, is_show=False):
        """

        :param img_path: 检测图片路径
        :param pretrained_model: 训练好的模型
        :param cuda: 是否使用GPU
        :param is_show: 显示图片
        """
        self.img_size = [94, 24]
        self.lpr_max_len = 8
        self.test_batch_size = 1
        self.num_workers = 1
        self.cuda = cuda
        self.is_show = is_show
        self.img_path = img_path
        self.pretrained_model = pretrained_model
        self.dropout_rate = 0
        self.phase_train = False

    def collate_fn(self, batch):
        imgs = []
        labels = []
        lengths = []
        for _, sample in enumerate(batch):
            img, label, length = sample
            imgs.append(torch.from_numpy(img))
            labels.extend(label)
            lengths.append(length)
        labels = np.asarray(labels).flatten().astype(np.float32)

        return torch.stack(imgs, 0), torch.from_numpy(labels), lengths

    def test(self):
        lprnet = build_lprnet(lpr_max_len=self.lpr_max_len, phase=self.phase_train, class_num=len(CHARS),
                              dropout_rate=self.dropout_rate)
        device = torch.device("cuda:0" if self.cuda else "cpu")
        lprnet.to(device)
        print("Successful to build network!")

        # load pretrained model
        if self.pretrained_model:
            lprnet.load_state_dict(torch.load(self.pretrained_model))
            print("load pretrained model successful!")
        else:
            print("[Error] Can't found pretrained mode, please check!")
            return False

        test_img_dirs = os.path.expanduser(self.img_path)
        transform = transforms.Grayscale(
            num_output_channels=3
        )
        test_dataset = LPRDataLoader(test_img_dirs, self.img_size, self.lpr_max_len, rgb2gray=transform)
        try:
            self.greedy_decode_eval(lprnet, test_dataset)
        finally:
            cv2.destroyAllWindows()

    def greedy_decode_eval(self, net, datasets):
        batch_iterator = DataLoader(datasets, self.test_batch_size, shuffle=False, num_workers=self.num_workers,
                                    collate_fn=self.collate_fn)
        Tp = 0
        Tn_1 = 0
        Tn_2 = 0
        t1 = time.time()
        # load train data
        images, labels, lengths = next(iter(batch_iterator))
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start + length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])
        imgs = images.numpy().copy()

        if self.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            result = ""
            for idx in label:
                result += CHARS[idx]
            print(result)
            # show image and its predict label
            if self.is_show:
                self.show(imgs[i], label, targets[i])
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1
        Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
        print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp + Tn_1 + Tn_2)))
        t2 = time.time()
        print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))

    def show(self, img, label, target):
        img = np.transpose(img, (1, 2, 0))
        img *= 128.
        img += 127.5
        img = img.astype(np.uint8)

        lb = ""
        for i in label:
            lb += CHARS[i]
        tg = ""
        for j in target.tolist():
            tg += CHARS[int(j)]

        flag = "F"
        if lb == tg:
            flag = "T"
        # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
        img = self.cv2ImgAddText(img, lb, (0, 0))
        cv2.imshow("test", img)
        print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb, "size: ", img.shape)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def cv2ImgAddText(self, img, text, pos, textColor=(255, 0, 0), textSize=12):
        if isinstance(img, np.ndarray):  # detect opencv format or not
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
        draw.text(pos, text, textColor, font=fontText)

        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# def get_parser():
#     parser = argparse.ArgumentParser(description='parameters to train net')
#     parser.add_argument('--img_size', default=[94, 24], help='the image size')
#     parser.add_argument('--test_img_dirs', default="data/predict/贵A406A2.jpg", help='the test images path')
#     parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
#     parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
#     parser.add_argument('--test_batch_size', default=1, help='testing batch size.')
#     parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
#     parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
#     parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
#     parser.add_argument('--show', default=True, type=bool, help='show test image and its predict result or not.')
#     parser.add_argument('--pretrained_model', default='./weights/LPRNet__iteration_8000.pth',
#                         help='pretrained base model')
#
#     args = parser.parse_args()
#
#     return args
#
#
# def collate_fn(batch):
#     imgs = []
#     labels = []
#     lengths = []
#     for _, sample in enumerate(batch):
#         img, label, length = sample
#         imgs.append(torch.from_numpy(img))
#         labels.extend(label)
#         lengths.append(length)
#     labels = np.asarray(labels).flatten().astype(np.float32)
#
#     return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)
#
#
# def test():
#     args = get_parser()
#
#     lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS),
#                           dropout_rate=args.dropout_rate)
#     device = torch.device("cuda:0" if args.cuda else "cpu")
#     lprnet.to(device)
#     print("Successful to build network!")
#
#     # load pretrained model
#     if args.pretrained_model:
#         lprnet.load_state_dict(torch.load(args.pretrained_model))
#         print("load pretrained model successful!")
#     else:
#         print("[Error] Can't found pretrained mode, please check!")
#         return False
#
#     test_img_dirs = os.path.expanduser(args.test_img_dirs)
#     transform = transforms.Grayscale(
#         num_output_channels=3
#     )
#     test_dataset = LPRDataLoader(test_img_dirs, args.img_size, args.lpr_max_len, rgb2gray=transform)
#     try:
#         Greedy_Decode_Eval(lprnet, test_dataset, args)
#     finally:
#         cv2.destroyAllWindows()
#
#
# def Greedy_Decode_Eval(Net, datasets, args):
#     # TestNet = Net.eval()
#     batch_iterator = DataLoader(datasets, args.test_batch_size, shuffle=False, num_workers=args.num_workers,
#                                 collate_fn=collate_fn)
#     Tp = 0
#     Tn_1 = 0
#     Tn_2 = 0
#     t1 = time.time()
#     # load train data
#     images, labels, lengths = next(iter(batch_iterator))
#     start = 0
#     targets = []
#     for length in lengths:
#         label = labels[start:start + length]
#         targets.append(label)
#         start += length
#     targets = np.array([el.numpy() for el in targets])
#     imgs = images.numpy().copy()
#
#     if args.cuda:
#         images = Variable(images.cuda())
#     else:
#         images = Variable(images)
#
#     # forward
#     prebs = Net(images)
#     # greedy decode
#     prebs = prebs.cpu().detach().numpy()
#     preb_labels = list()
#     for i in range(prebs.shape[0]):
#         preb = prebs[i, :, :]
#         preb_label = list()
#         for j in range(preb.shape[1]):
#             preb_label.append(np.argmax(preb[:, j], axis=0))
#         no_repeat_blank_label = list()
#         pre_c = preb_label[0]
#         if pre_c != len(CHARS) - 1:
#             no_repeat_blank_label.append(pre_c)
#         for c in preb_label:  # dropout repeate label and blank label
#             if (pre_c == c) or (c == len(CHARS) - 1):
#                 if c == len(CHARS) - 1:
#                     pre_c = c
#                 continue
#             no_repeat_blank_label.append(c)
#             pre_c = c
#         preb_labels.append(no_repeat_blank_label)
#     for i, label in enumerate(preb_labels):
#         # show image and its predict label
#         if args.show:
#             show(imgs[i], label, targets[i])
#         if len(label) != len(targets[i]):
#             Tn_1 += 1
#             continue
#         if (np.asarray(targets[i]) == np.asarray(label)).all():
#             Tp += 1
#         else:
#             Tn_2 += 1
#     Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
#     print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp + Tn_1 + Tn_2)))
#     t2 = time.time()
#     print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))
#
#
# def show(img, label, target):
#     img = np.transpose(img, (1, 2, 0))
#     img *= 128.
#     img += 127.5
#     img = img.astype(np.uint8)
#
#     lb = ""
#     for i in label:
#         lb += CHARS[i]
#     tg = ""
#     for j in target.tolist():
#         tg += CHARS[int(j)]
#
#     flag = "F"
#     if lb == tg:
#         flag = "T"
#     # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
#     img = cv2ImgAddText(img, lb, (0, 0))
#     cv2.imshow("test", img)
#     print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb, "size: ", img.shape)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
#
# def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
#     if (isinstance(img, np.ndarray)):  # detect opencv format or not
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(img)
#     fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
#     draw.text(pos, text, textColor, font=fontText)
#
#     return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    detect = ReadLicense(img_path="data/predict/贵A406A2.jpg", pretrained_model="./weights/LPRNet__iteration_8000.pth",
                         cuda=True, is_show=False)
    detect.test()
