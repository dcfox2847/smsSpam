import sys

from data import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, FigureCanvasQTAgg
from sklearn.model_selection import train_test_split
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot, QAbstractTableModel, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTableView
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

class Data_Table(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt. Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


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
        # The 'data' variable, will hold all of the data instantiated from teh data, class
        self.data = Data()
        self.ui.check_message_button.clicked.connect(self.message_button_clicked)
        hist_chart = Hist_Canvas(self.ui.hist_widget)
        heat_chart = Heat_Canvas(self.ui.heat_widget)
        train_word_cloud_chart = Word_Cloud_Canvas(self.ui.train_cloud_widget, "Train")
        test_word_cloud_chart = Word_Cloud_Canvas(self.ui.test_cloud_widget, "Test")
        # Add the metric information to the labels on the main page of the application
        acc_text = self.ui.accuracy_label.text() + " " + self.data.accuracy_score
        prec_text = self.ui.precision_label.text() + " " + self.data.precision_score
        recall_text = self.ui.recall_label.text() + " " + self.data.recall_score
        f1_text = self.ui.f1_label.text() + " " + self.data.f1_score
        self.ui.accuracy_label.setText(acc_text)
        self.ui.precision_label.setText(prec_text)
        self.ui.recall_label.setText(recall_text)
        self.ui.f1_label.setText(f1_text)
        data_table = Data_Table(self.data.new_test_dataframe)
        self.ui.data_set_table.setModel(data_table)
        self.ui.data_set_table.resizeColumnsToContents()


    def message_button_clicked(self):
        # data = Data()
        message_test = self.ui.message_text_field.text()
        text = self.data.count_vector.transform([message_test])
        prediction = self.data.mnb.predict(text)
        show_prediction = str(prediction)
        characters_to_remove = "',[,]"
        for character in characters_to_remove:
            show_prediction = show_prediction.replace(character, '')
        if show_prediction == '0':
            final_string = "Ham"
        elif show_prediction == '1':
            final_string = "Spam"
        else:
            final_string = "Human checking is needed at this time for validation."
        self.ui.result_label.setText(final_string)

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