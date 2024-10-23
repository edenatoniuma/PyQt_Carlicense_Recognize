# -*- coding: utf-8 -*-
# /usr/bin/env/python3

"""
test pretrained model.
Author: aiboy.wei@outlook.com .
"""
import yaml

from detect_model.data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from detect_model.model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
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


class ReadLicensePlate:
    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        self.config = self.load_config()
        self.img_size = self.config.get("img_size")
        self.predict_dir = self.config.get("predict_dir")
        self.lpr_max_len = self.config.get("lpr_max_len")
        self.batch_size = self.config.get("batch_size")
        self.num_workers = self.config.get("num_workers")
        self.cuda = self.config.get("cuda")
        self.weights = self.config.get("weights")

    def load_config(self):
        with open(self.yaml_file, "r") as file:
            config = yaml.safe_load(file)
        return config

    def get_parser(self):
        parser = argparse.ArgumentParser(description='parameters to train net')
        parser.add_argument('--img_size', default=self.img_size, help='the image size')
        parser.add_argument('--test_img_dirs', default=self.predict_dir, help='the test images path')
        parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
        parser.add_argument('--lpr_max_len', default=self.lpr_max_len, help='license plate number max length.')
        parser.add_argument('--test_batch_size', default=self.batch_size, help='testing batch size.')
        parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
        parser.add_argument('--num_workers', default=self.num_workers, type=int,
                            help='Number of workers used in dataloading')
        parser.add_argument('--cuda', default=self.cuda, type=bool, help='Use cuda to train model')
        parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
        parser.add_argument('--pretrained_model', default=self.weights,
                            help='pretrained base model')

        args = parser.parse_args()

        return args

    @staticmethod
    def collate_fn(batch):
        imgs = []
        labels = []
        lengths = []
        for _, sample in enumerate(batch):
            img, label, length = sample
            imgs.append(torch.from_numpy(img))
            labels.extend(label)
            lengths.append(length)
        labels = np.asarray(labels).flatten().astype(np.float32)

        return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

    def test(self):
        args = self.get_parser()

        lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS),
                              dropout_rate=args.dropout_rate)
        device = torch.device("cuda:0" if args.cuda else "cpu")
        lprnet.to(device)
        # print("Successful to build network!")

        # load pretrained model
        if args.pretrained_model:
            lprnet.load_state_dict(torch.load(args.pretrained_model))
            # print("load pretrained model successful!")
        else:
            print("[Error] Can't found pretrained mode, please check!")
            return False

        test_img_dirs = os.path.expanduser(args.test_img_dirs)
        # transform = transforms.Grayscale(
        #     num_output_channels=3
        # )
        test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
        try:
            result = self.Greedy_Decode_Eval(lprnet, test_dataset, args)
        finally:
            cv2.destroyAllWindows()
        return result

    def Greedy_Decode_Eval(self, Net, datasets, args):
        # TestNet = Net.eval()
        epoch_size = len(datasets) // args.test_batch_size
        batch_iterator = iter(
            DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers,
                       collate_fn=self.collate_fn))

        result = ""
        t1 = time.time()
        for i in range(epoch_size):
            # load train data
            images, labels, lengths = next(batch_iterator)
            start = 0
            targets = []
            for length in lengths:
                label = labels[start:start + length]
                targets.append(label)
                start += length

            if args.cuda:
                images = Variable(images.cuda())
            else:
                images = Variable(images)

            # forward
            prebs = Net(images)
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
                for label in preb_labels[0]:
                    result += CHARS[label]
        t2 = time.time()
        return f"[Info] Test Speed: {format((t2 - t1) / len(datasets))}s\n{result}"


if __name__ == "__main__":
    predict = ReadLicensePlate("config.yaml")
    print(predict.test())
