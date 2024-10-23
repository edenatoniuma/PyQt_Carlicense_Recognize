#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :call_show_logger.py
# @Time      :2024/6/24 10:51
# @Author    :嘉隆
import os
import sys

from login_register.show_logger import Ui_Log
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget, QStatusBar, QLineEdit


class LogWindow(QWidget, Ui_Log):
    def __init__(self, parent=None, log_file="logs/app.log", is_admin=0):
        super(LogWindow, self).__init__(parent)
        self.choice_window = None
        self.setupUi(self)
        self.is_admin = is_admin
        self.log_file = log_file
        self.setWindowTitle("LogInfo")
        self.show_log()
        self.back_btn.clicked.connect(self.back_choice_window)
        self.show()

    def show_log(self):
        if not os.path.exists(self.log_file):
            QMessageBox.Critical(self, "file error", "wrong log file path!")
        else:
            with open(self.log_file, 'r') as f:
                log_content = f.read()
                self.log_blank.setText(log_content)

    def back_choice_window(self):
        from call_choice_fun import ChoiceFunWindow
        self.choice_window = ChoiceFunWindow(is_admin=self.is_admin)
        self.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LogWindow(log_file="logs/app.log")
    sys.exit(app.exec_())
