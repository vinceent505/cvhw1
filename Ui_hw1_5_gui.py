# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\User\Desktop\Hw1\cvhw1\hw1_5_gui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(312, 545)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(19, 20, 271, 461))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label, 0, QtCore.Qt.AlignTop)
        self.pushButton_5_1 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_5_1.setObjectName("pushButton_5_1")
        self.verticalLayout.addWidget(self.pushButton_5_1, 0, QtCore.Qt.AlignTop)
        self.pushButton_5_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_5_2.setObjectName("pushButton_5_2")
        self.verticalLayout.addWidget(self.pushButton_5_2, 0, QtCore.Qt.AlignTop)
        self.pushButton_5_3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_5_3.setObjectName("pushButton_5_3")
        self.verticalLayout.addWidget(self.pushButton_5_3, 0, QtCore.Qt.AlignTop)
        self.pushButton_5_4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_5_4.setObjectName("pushButton_5_4")
        self.verticalLayout.addWidget(self.pushButton_5_4, 0, QtCore.Qt.AlignTop)
        self.spinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.spinBox.setObjectName("spinBox")
        self.verticalLayout.addWidget(self.spinBox, 0, QtCore.Qt.AlignTop)
        self.pushButton_5_5 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_5_5.setObjectName("pushButton_5_5")
        self.verticalLayout.addWidget(self.pushButton_5_5, 0, QtCore.Qt.AlignTop)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 312, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "VGG16 TEST"))
        self.pushButton_5_1.setText(_translate("MainWindow", "1.Show Train Images"))
        self.pushButton_5_2.setText(_translate("MainWindow", "2. Show HyperParameters"))
        self.pushButton_5_3.setText(_translate("MainWindow", "3. Show Model Shortcut"))
        self.pushButton_5_4.setText(_translate("MainWindow", "4. Show Accuracy"))
        self.pushButton_5_5.setText(_translate("MainWindow", "5. Test"))

