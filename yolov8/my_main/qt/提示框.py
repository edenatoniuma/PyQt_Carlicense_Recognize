#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :提示框.py
# @Time      :2024/6/13 16:21
# @Author    :嘉隆
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QMainWindow, QToolTip
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QSize, Qt


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        QToolTip.setFont(QFont('SansSerif', 10))
        self.setToolTip('this is a <b>QWidget</b> widget')
        btn = QPushButton("button", self)
        btn.setToolTip("this is a <b>QPushButton</b> widget")
        btn.resize(btn.sizeHint())
        btn.move(50, 50)
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle("ToolTips")
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
