#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :call_login.py
# @Time      :2024/6/14 15:54
# @Author    :嘉隆
import logging
import sys

import pymysql
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget, QStatusBar, QLineEdit
from login_register.login import Ui_Form
from PyQt5.QtCore import pyqtSlot, QEventLoop, QTimer
from utils import setup_logging
import warnings

warnings.filterwarnings("ignore")
setup_logging()
logger = logging.getLogger("log")


class LoginWindow(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(LoginWindow, self).__init__(parent)
        self.choice_fun_window = None
        logger.info("Application started")
        self.detect_window = None
        self.cursor = None
        self.connection = None
        self.register_window = None
        # 是否为管理员
        self.is_admin = False
        # 初始化UI
        self.setupUi(self)
        # 按钮初始化
        self.login_in_btn.clicked.connect(self.login_in)
        self.sign_up_btn.clicked.connect(self.open_register_window)
        self.exit_btn.clicked.connect(self.close)
        # 数据库初始化
        self.init_db()
        # 设置密码输入栏格式
        self.account_edit.setPlaceholderText("Enter Your Account")
        self.pwd_edit.setPlaceholderText("Enter Your Password")
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        # 显示窗口
        self.show()

    # 连接数据库
    def init_db(self):
        """
        连接到mysql数据库，为后面登录验证账号做准备
        :return:
        """
        try:
            self.connection = pymysql.connect(
                host="localhost",
                user="root",
                password="134679",
                db="pyqt_user",
                charset="utf8",
                cursorclass=pymysql.cursors.DictCursor
            )
            self.cursor = self.connection.cursor()
            logger.info("initialize database successful")
            QMessageBox.information(self, "connection", "Database connection successful")
        except pymysql.MySQLError as e:
            logger.critical(f"Error: {e}")
            QMessageBox.critical(self, "connection", f"Database connection failed : {e}")
            sys.exit()

    @pyqtSlot()
    def login_in(self):
        """
        用上面定义的self.cursor游标来执行 sql命令，查看账号与密码是否匹配数据库中的内容
        :return:
        """
        username = self.account_edit.text()
        password = self.pwd_edit.text()
        if not username or not password:
            QMessageBox.warning(self, "Input Error", "Username and Password cannot be empty!")
            return

        try:
            sql = "SELECT * FROM users WHERE username = %s AND password = %s"
            self.cursor.execute(sql, (username, password))
            result = self.cursor.fetchone()
            if result:
                self.is_admin = result['is_admin']
                self.text_blank.setText(f"登录成功！\n{username}")
                # 打开功能界面：用户管理、日志查看、检测
                QTimer.singleShot(2000, self.open_choice_fun_window)
            else:
                self.text_blank.setText(f"登录失败！\n用户名或密码错误")
        except pymysql.MySQLError as e:
            QMessageBox.critical(self, "Database Error", f"Failed to login in: {e}")

    def open_register_window(self):
        from call_register import RegisterWindow
        logger.info("open register window successful")
        self.register_window = RegisterWindow()
        self.hide()

    # def open_detect_window(self):
    #     from call_detect import DetectWindow
    #     logger.info("open detect window successfully")
    #     self.detect_window = DetectWindow()
    #     self.hide()

    def open_choice_fun_window(self):
        from call_choice_fun import ChoiceFunWindow
        logger.info("open choice function window successful")
        self.choice_fun_window = ChoiceFunWindow(is_admin=self.is_admin)
        self.hide()

    def return_admin_state(self):
        return self.is_admin

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '确认关闭', '你确定要关闭程序吗?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            logger.info("App shutdown")
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_form = LoginWindow()
    sys.exit(app.exec_())
