import sys

from matplotlib.figure import Figure
import dataCleaning as dc
import wordCloudGen as wcg
from data import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, FigureCanvasQTAgg
from sklearn.model_selection import train_test_split
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QMainWindow
from MainScreen import Ui_MainWindow

# matplotlib.use("Qt5Agg")

class Hist_Canvas(FigureCanvas):
    def __init__(self, parent):
        fig, self.ax1 = plt.subplots(nrows=1, ncols=2, tight_layout=True, figsize=(9, 6))
        data = Data()
        super().__init__(fig)
        self.setParent(parent)
        data.df2.hist(column="Length", by="Label", bins=50, ax=self.ax1)
        plt.show()

class Heat_Canvas(FigureCanvas):
    def __init__(self, parent):
        fig, self.ax1 = plt.subplots(nrows=1, ncols=1, tight_layout=True, figsize=(9, 6))
        data = Data()
        x_train, x_test, y_train, y_test = train_test_split(data.df['SMS'], data.df['Label'], test_size=0.20,
                                                            random_state=1)
        super().__init__(fig)
        self.setParent(parent)
        conf_matrix = confusion_matrix(y_test, data.predictions)
        df_cm = pd.DataFrame(conf_matrix, index = data.mnb.classes_,
                             columns=data.mnb.classes_)
        sns.heatmap(df_cm, annot=True, fmt="d", ax=self.ax1)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()


class Word_Cloud_Canvas(FigureCanvas):
    def __init__(self, parent, type):
        fig, self.ax1 = plt.subplots(nrows=1, ncols=1, tight_layout=True, figsize=(9, 6))
        data = Data()
        super().__init__(fig)
        self.setParent(parent)
        training_frames, test_frames = dataCleaning.make_sets(data.df, data.training_frames, data.test_frames)
        stop_loop = False
        while not stop_loop:
            if type.lower() == "train":
                wordCloudGen.show_wordcloud(training_frames, "Training Set")
                stop_loop = True
            elif type.lower() == "test":
                wordCloudGen.show_wordcloud(test_frames, "Test Set")
                stop_loop = True
            else:
                print("Please enter either 'Test' or 'Train' as the argmuent....")

        plt.show()
        # wordCloudGen.show_wordcloud_alt(test_frames, "Test Set", self.ax1)


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

        # BELOW IS TESTING THAT DATA CAN BE ACCESSED FROM INSIDE THE CLASS ONCE INSTANTIATED
        # The 'data' variable, will hold all of the data instantiated from teh data, class
        data = Data()
        hist_chart = Hist_Canvas(self.ui.hist_widget)
        heat_chart = Heat_Canvas(self.ui.heat_widget)
        train_word_cloud_chart = Word_Cloud_Canvas(self.ui.train_cloud_widget, "Train")
        test_word_cloud_chart = Word_Cloud_Canvas(self.ui.test_cloud_widget, "Test")


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