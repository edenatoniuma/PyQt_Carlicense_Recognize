#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :窗口居中.py
# @Time      :2024/6/14 15:08
# @Author    :嘉隆

import sys
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.resize(250, 150)
        self.center()
        self.setWindowTitle("center")
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
