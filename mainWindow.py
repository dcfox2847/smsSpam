import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow

from MainScreen import Ui_MainWindow


class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)

        self.ui.stackedWidget.setCurrentWidget(self.ui.data_page)
        self.ui.data_button.clicked.connect(self.show_data)
        self.ui.hist_button.clicked.connect(self.show_hist)
        self.ui.heat_button.clicked.connect(self.show_heat)
        self.ui.train_cloud_button.clicked.connect(self.show_train_cloud)
        self.ui.test_cloud_button.clicked.connect(self.show_test_cloud)
        self.ui.check_button.clicked.connect(self.show_check_message)

    def show(self):
        self.main_win.show()

    def show_data(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.data_page)

    def show_hist(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.hist_page)

    def show_heat(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.heat_page)

    def show_train_cloud(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.train_cloud_page)

    def show_test_cloud(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.test_cloud_page)

    def show_check_message(self):
        self.ui.stackedWidget.setCurrentWidget((self.ui.check_page))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())