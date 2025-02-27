# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'detect.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(990, 846)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.detect_object_text = QtWidgets.QTextBrowser(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.detect_object_text.sizePolicy().hasHeightForWidth())
        self.detect_object_text.setSizePolicy(sizePolicy)
        self.detect_object_text.setObjectName("detect_object_text")
        self.gridLayout.addWidget(self.detect_object_text, 0, 1, 1, 1)
        self.predict_result_label = QtWidgets.QLabel(Form)
        self.predict_result_label.setMinimumSize(QtCore.QSize(495, 270))
        self.predict_result_label.setObjectName("predict_result_label")
        self.gridLayout.addWidget(self.predict_result_label, 0, 0, 1, 1)
        self.original_img_label = QtWidgets.QLabel(Form)
        self.original_img_label.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.original_img_label.sizePolicy().hasHeightForWidth())
        self.original_img_label.setSizePolicy(sizePolicy)
        self.original_img_label.setMinimumSize(QtCore.QSize(495, 270))
        self.original_img_label.setObjectName("original_img_label")
        self.gridLayout.addWidget(self.original_img_label, 1, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.load_gif_label = QtWidgets.QLabel(Form)
        self.load_gif_label.setMaximumSize(QtCore.QSize(64, 161))
        self.load_gif_label.setObjectName("load_gif_label")
        self.verticalLayout.addWidget(self.load_gif_label, 0, QtCore.Qt.AlignHCenter)
        self.detect_btn = QtWidgets.QPushButton(Form)
        self.detect_btn.setObjectName("detect_btn")
        self.verticalLayout.addWidget(self.detect_btn)
        self.select_img_btn = QtWidgets.QPushButton(Form)
        self.select_img_btn.setObjectName("select_img_btn")
        self.verticalLayout.addWidget(self.select_img_btn)
        self.back_btn = QtWidgets.QPushButton(Form)
        self.back_btn.setObjectName("back_btn")
        self.verticalLayout.addWidget(self.back_btn)
        self.gridLayout.addLayout(self.verticalLayout, 1, 1, 1, 1)
        self.read_license_plate_label = QtWidgets.QLabel(Form)
        self.read_license_plate_label.setMinimumSize(QtCore.QSize(495, 270))
        self.read_license_plate_label.setObjectName("read_license_plate_label")
        self.gridLayout.addWidget(self.read_license_plate_label, 3, 0, 1, 1)
        self.read_license_plate_text = QtWidgets.QTextBrowser(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.read_license_plate_text.sizePolicy().hasHeightForWidth())
        self.read_license_plate_text.setSizePolicy(sizePolicy)
        self.read_license_plate_text.setObjectName("read_license_plate_text")
        self.gridLayout.addWidget(self.read_license_plate_text, 3, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.predict_result_label.setText(_translate("Form", "检测结果"))
        self.original_img_label.setText(_translate("Form", "原始图片"))
        self.load_gif_label.setText(_translate("Form", "TextLabel"))
        self.detect_btn.setText(_translate("Form", "检测"))
        self.select_img_btn.setText(_translate("Form", "上传图片"))
        self.back_btn.setText(_translate("Form", "返回"))
        self.read_license_plate_label.setText(_translate("Form", "检测车牌"))
