#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :call_choice_fun.py
# @Time      :2024/6/24 11:08
# @Author    :嘉隆

import sys
import logging
from utils import setup_logging
from login_register.choice_fun import Ui_Function
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget, QStatusBar, QLineEdit
from PyQt5.QtCore import pyqtSlot, QEventLoop, QTimer

setup_logging()
logger = logging.getLogger("choice_fun_log")


class ChoiceFunWindow(QWidget, Ui_Function):
    def __init__(self, parent=None, log_file=None, is_admin=0):
        super(ChoiceFunWindow, self).__init__(parent)
        self.manage_user_window = None
        self.setupUi(self)
        self.log_window = None
        self.detect_window = None
        self.is_admin = is_admin
        # 按钮初始化
        self.detect_fun_btn.clicked.connect(self.open_detect_window)
        # 非管理员不可用
        self.read_logger_btn.setEnabled(False)
        self.manage_user_btn.setEnabled(False)
        self.judge_is_admin()
        self.read_logger_btn.clicked.connect(self.open_log_window)
        self.manage_user_btn.clicked.connect(self.open_manage_user_window)
        self.show()

    def open_detect_window(self):
        from call_detect import DetectWindow
        logger.info("open detect window successful")
        self.detect_window = DetectWindow(is_admin=self.is_admin)
        self.hide()

    def open_log_window(self):
        from call_show_logger import LogWindow
        logger.info("check log successful")
        self.log_window = LogWindow(is_admin=self.is_admin)
        self.hide()
    
    def open_manage_user_window(self):
        from call_manage_user import ManageWindow
        logger.info("manage user function started")
        self.manage_user_window = ManageWindow(is_admin=self.is_admin)
        self.hide()

    def judge_is_admin(self):
        if self.is_admin == 1:
            self.read_logger_btn.setEnabled(True)
            self.manage_user_btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChoiceFunWindow(log_file="logs/app.log")
    sys.exit(app.exec_())
