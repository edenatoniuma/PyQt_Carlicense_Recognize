# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'show_logger.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Log(object):
    def setupUi(self, Log):
        Log.setObjectName("Log")
        Log.resize(1100, 809)
        self.gridLayout = QtWidgets.QGridLayout(Log)
        self.gridLayout.setContentsMargins(30, 30, 30, 30)
        self.gridLayout.setObjectName("gridLayout")
        self.log_blank = QtWidgets.QTextBrowser(Log)
        self.log_blank.setObjectName("log_blank")
        self.gridLayout.addWidget(self.log_blank, 0, 0, 1, 1)
        self.back_btn = QtWidgets.QPushButton(Log)
        self.back_btn.setObjectName("back_btn")
        self.gridLayout.addWidget(self.back_btn, 1, 0, 1, 1)

        self.retranslateUi(Log)
        QtCore.QMetaObject.connectSlotsByName(Log)

    def retranslateUi(self, Log):
        _translate = QtCore.QCoreApplication.translate
        Log.setWindowTitle(_translate("Log", "Form"))
        self.log_blank.setHtml(_translate("Log", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.back_btn.setText(_translate("Log", "返回"))
