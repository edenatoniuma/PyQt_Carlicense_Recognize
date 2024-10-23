#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :关闭窗口.py
# @Time      :2024/6/14 11:12
# @Author    :嘉隆
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QMainWindow, QToolTip
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QSize, QCoreApplication


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        qbtn = QPushButton('quit', self)
        qbtn.clicked.connect(QCoreApplication.instance().quit)
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(50, 50)

        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle("Quit button")
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
