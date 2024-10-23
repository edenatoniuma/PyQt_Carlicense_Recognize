#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :call_register.py
# @Time      :2024/6/17 8:49
# @Author    :嘉隆

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import pyqtSlot
from login_register.register import Ui_Form

import warnings
import pymysql

warnings.filterwarnings("ignore")


class RegisterWindow(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(RegisterWindow, self).__init__(parent)
        self.cursor = None
        self.connection = None
        self.login_window = None
        self.setupUi(self)
        # 跳转到登录
        self.login_in_btn.clicked.connect(self.open_login_window)
        self.sign_up_btn.clicked.connect(self.sign_up)
        self.exit_btn.clicked.connect(self.close)
        # 提示
        self.account_edit.setPlaceholderText("enter you account/name long:0-255")
        self.pwd_edit.setPlaceholderText("enter you password long:0-255")
        self.init_db()
        self.show()

    def display(self):
        username = self.account_edit.text()
        self.text_blank.setText(f"注册成功！\n{username}")

    def init_db(self):
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
            QMessageBox.information(self, "connection", "Database connection successful")
        except pymysql.MySQLError as e:
            QMessageBox.critical(self, "connection", f"Database connection failed : {e}")
            sys.exit()

    @pyqtSlot()
    def sign_up(self):
        username = self.account_edit.text()
        password = self.pwd_edit.text()
        confirm_password = self.confirm_pwd_edit.text()
        if not username or not password:
            QMessageBox.warning(self, "Input Error", "Username and Password cannot be empty!")
            return
        if password != confirm_password:
            QMessageBox.warning(self, "Input Error", "Password and Confirm_Password dont same!")
            return
        try:
            sql = "INSERT INTO users (username, password) VALUES (%s, %s)"
            self.cursor.execute(sql, (username, password))
            self.connection.commit()
            self.text_blank.setText(f"注册成功！\n{username}")
        except pymysql.MySQLError as e:
            QMessageBox.critical(self, 'Database Error', f'Failed to register user: {e}')
            self.connection.rollback()

    def open_login_window(self):
        from call_login import LoginWindow
        self.login_window = LoginWindow(self)
        self.hide()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '确认关闭', '你确定要关闭程序吗?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_form = RegisterWindow()
    sys.exit(app.exec_())
