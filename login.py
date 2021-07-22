# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'login.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(480, 640)
        Dialog.setStyleSheet("background-color: rgb(54, 54, 54);")
        self.login_label = QtWidgets.QLabel(Dialog)
        self.login_label.setGeometry(QtCore.QRect(170, 40, 111, 61))
        font = QtGui.QFont()
        font.setPointSize(28)
        self.login_label.setFont(font)
        self.login_label.setStyleSheet("color: #ff007f;")
        self.login_label.setAlignment(QtCore.Qt.AlignCenter)
        self.login_label.setObjectName("login_label")
        self.username_label = QtWidgets.QLabel(Dialog)
        self.username_label.setGeometry(QtCore.QRect(40, 200, 101, 31))
        self.username_label.setStyleSheet("font-size:15pt;\n"
"color: #ff007f;")
        self.username_label.setObjectName("username_label")
        self.password_label = QtWidgets.QLabel(Dialog)
        self.password_label.setGeometry(QtCore.QRect(40, 290, 101, 31))
        self.password_label.setStyleSheet("font-size:15pt;\n"
"color: #ff007f;")
        self.password_label.setObjectName("password_label")
        self.username_field = QtWidgets.QLineEdit(Dialog)
        self.username_field.setGeometry(QtCore.QRect(180, 200, 241, 31))
        self.username_field.setStyleSheet("border: 1px solid white;\n"
"font-size: 14pt;\n"
"color: #f3f3f3;")
        self.username_field.setInputMask("")
        self.username_field.setObjectName("username_field")
        self.password_field = QtWidgets.QLineEdit(Dialog)
        self.password_field.setGeometry(QtCore.QRect(180, 290, 241, 31))
        self.password_field.setStyleSheet("border: 1px solid white;\n"
"font-size: 14pt;\n"
"color: #f3f3f3;")
        self.password_field.setInputMask("")
        self.password_field.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_field.setReadOnly(False)
        self.password_field.setObjectName("password_field")
        self.login_button = QtWidgets.QPushButton(Dialog)
        self.login_button.setGeometry(QtCore.QRect(330, 350, 131, 51))
        self.login_button.setStyleSheet("background-color: rgb(211, 215, 207);\n"
"font-size: 18px;")
        self.login_button.setObjectName("login_button")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.login_label.setText(_translate("Dialog", "Login"))
        self.username_label.setText(_translate("Dialog", "Username"))
        self.password_label.setText(_translate("Dialog", "Password"))
        self.login_button.setText(_translate("Dialog", "Login"))

