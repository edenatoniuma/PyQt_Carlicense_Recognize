#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :call_manage_user.py
# @Time      :2024/6/24 15:28
# @Author    :嘉隆


import sys
import logging

import pymysql

from utils import setup_logging
from login_register.manage_user import Ui_Form
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget, QStatusBar, QLineEdit, QTableWidgetItem
from PyQt5.QtCore import pyqtSlot, QEventLoop, QTimer

setup_logging()
logger = logging.getLogger("manage_user_log")


class ManageWindow(QWidget, Ui_Form):
    def __init__(self, parent=None, is_admin=0):
        super(ManageWindow, self).__init__(parent)

        self.choice_window = None
        self.connection = None
        self.cursor = None
        self.is_admin = is_admin
        # UI初始化
        self.setupUi(self)
        # 连接数据库
        self.init_db()
        # 加载内容
        self.load_data()
        # 添加提示
        self.username_edit.setPlaceholderText("username")
        self.password_edit.setPlaceholderText("password")
        self.is_admin_edit.setPlaceholderText("is_admin")
        # 按钮初始化
        self.back_btn.clicked.connect(self.back_choice_window)
        self.add_btn.clicked.connect(self.add_user)
        self.delete_btn.clicked.connect(self.delete_user)
        self.update_btn.clicked.connect(self.update_user)
        self.search_btn.clicked.connect(self.search_user)
        # 展示页面
        self.show()

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

    # 显示数据
    def load_data(self):
        try:
            sql = "SELECT * FROM users"
            self.cursor.execute(sql)
            rows = self.cursor.fetchall()
            # 设置行
            self.user_table.setRowCount(len(rows))
            # 设置列
            self.user_table.setColumnCount(len(rows[0]))
            # 设置列名
            self.user_table.setHorizontalHeaderLabels(list(rows[0].keys()))
            for row_idx, row_data in enumerate(rows):
                for col_idx, (key, value) in enumerate(row_data.items()):
                    self.user_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
            logger.info("load data successful")
        except pymysql.MySQLError as e:
            QMessageBox.critical(self, "Database Error", f"Failed to load data in :{e}")

    # 查
    def search_user(self):
        try:
            name = self.username_edit.text()
            self.cursor.execute("SELECT * FROM users WHERE username LIKE %s", ("%" + name + "%",))
            rows = self.cursor.fetchall()
            # 设置行
            self.user_table.setRowCount(len(rows))
            # 设置列
            self.user_table.setColumnCount(len(rows[0]))
            # 设置列名
            self.user_table.setHorizontalHeaderLabels(list(rows[0].keys()))
            for row_idx, row_data in enumerate(rows):
                for col_idx, (key, value) in enumerate(row_data.items()):
                    self.user_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
            logger.info(f"search user:{name}")
        except pymysql.MySQLError as e:
            QMessageBox.critical(self, "Database Error", f"Failed to load data in :{e}")

    # 改
    def update_user(self):
        try:
            selected_row = self.user_table.currentRow()
            if selected_row == -1:
                QMessageBox.warning(self, 'Error', 'select a user to update')
                return
            user_id = self.user_table.item(selected_row, 0).text()
            name = self.username_edit.text()
            password = self.password_edit.text()
            is_admin = self.is_admin_edit.text()
            if not (name and password and is_admin):
                QMessageBox.warning(self, 'Error', f'All fields are required')
                return
            self.cursor.execute('UPDATE users SET username = %s, password = %s, is_admin = %s WHERE id = %s',
                                (name, password, is_admin, user_id))
            self.connection.commit()
            self.load_data()
        except pymysql.MySQLError as e:
            QMessageBox.critical(self, "Database Error", f"Failed to load data in :{e}")
            self.connection.rollback()

    #   增
    def add_user(self):
        try:
            name = self.username_edit.text()
            password = self.password_edit.text()
            is_admin = self.is_admin_edit.text()
            if not (name and password and is_admin):
                QMessageBox.warning(self, 'Error', f'All fields are required')
                return
            self.cursor.execute('INSERT INTO users (username, password, is_admin) VALUES (%s, %s, %s)',
                                (name, password, is_admin))
            self.connection.commit()
            self.load_data()
        except pymysql.MySQLError as e:
            QMessageBox.critical(self, "Database Error", f"Failed to load data in :{e}")
            self.connection.rollback()

    def delete_user(self):
        try:
            selected_row = self.user_table.currentRow()
            if selected_row == -1:
                QMessageBox.warning(self, 'Error', 'Select a user to delete')
                return

            user_id = self.user_table.item(selected_row, 0).text()
            self.cursor.execute('DELETE FROM users WHERE id = %s', (user_id,))
            self.connection.commit()
            self.load_data()
        except pymysql.MySQLError as e:
            QMessageBox.critical(self, "Database Error", f"Failed to load data in :{e}")
            self.connection.rollback()

    def back_choice_window(self):
        from call_choice_fun import ChoiceFunWindow
        self.choice_window = ChoiceFunWindow(is_admin=self.is_admin)
        self.hide()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ManageWindow()
    sys.exit(app.exec_())
