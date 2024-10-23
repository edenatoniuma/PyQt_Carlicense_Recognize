#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :utils.py
# @Time      :2024/6/24 9:33
# @Author    :嘉隆

import logging
import os


def setup_logging():
    log_dir = "logs"
    logger = logging.getLogger("log")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 只有在没有处理器时才设置basicConfig
    if not logger.handlers:
        logging.basicConfig(
            filename=os.path.join(log_dir, "app.log"),
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

