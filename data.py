import wordCloudGen
from dataCleaning import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
import math
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import nltk
import warnings
import scipy as sp

class Data:
    # Variables
    column_names = ["label", "sms"]
    training_set = pd.DataFrame(columns=column_names)
    test_set = pd.DataFrame(columns=column_names)
    training_frames = pd.DataFrame(columns=column_names)
    test_frames = pd.DataFrame(columns=column_names)
    new_test_dataframe = pd.DataFrame(columns=['SMS', 'Length', 'Result'])
    i = 0

    def __init__(self):
        # MAIN BODY OF APPLICATION
        # Variables:
        self.column_names = ["label", "sms"]
        self.training_set = pd.DataFrame(columns=self.column_names)
        self.test_set = pd.DataFrame(columns=self.column_names)
        self.training_frames = pd.DataFrame(columns=self.column_names)
        self.test_frames = pd.DataFrame(columns=self.column_names)
        self.new_test_dataframe = pd.DataFrame(columns=['SMS', 'Length', 'Result'])
        self.i = 0
        self.mnb = MultinomialNB()
        self.predictions = None
        self.df = None
        self.df2 = None
        self.count_vector = None

        # Import the data from the dataset
        file_path = '/home/televator/Coding/spam_sample/smsspamcollection/SMSSpamCollection'
        # Take data from CSV dataset and put it into a Pandas DataFrame
        self.df = pd.read_csv(file_path, sep='\t', header=None, names=['Label', 'SMS'])
        self.df['Length'] = self.df['SMS'].apply(len)
        self.df2 = pd.read_csv(file_path, sep='\t', header=None, names=['Label', 'SMS'])
        self.df2['Length'] = self.df['SMS'].apply(len)
        print(self.df2)
        # Create keys for the labels
        self.df.loc[:, 'Label'] = self.df.Label.map({'ham':0, 'spam':1})
        # Split the data, and initiate the count vectorizer
        x_train, x_test, y_train, y_test = train_test_split(self.df['SMS'], self.df['Label'], test_size=0.20, random_state=1)
        # Create the count vectorizer
        self.count_vector = CountVectorizer()
        # Vectorize the data, then test it in the 'mnb' algorithm
        training_data = self.count_vector.fit_transform(x_train)
        testing_data = self.count_vector.transform(x_test)
        self.mnb = MultinomialNB()
        self.mnb.fit(training_data, y_train)
        self.predictions = self.mnb.predict(testing_data)
        # Create a new dataframe that will hold all result data for plotting
        # This data frame has the SMS message, the length of the message, and will show 'Spam' or 'Ham' as a result
        for row in x_test:
            message = row
            text = self.count_vector.transform([message])
            prediction = self.mnb.predict(text)
            new_string = str(prediction)
            characters_to_remove = "',[,]"
            for character in characters_to_remove:
                new_string = new_string.replace(character, '')
            if new_string == '0':
                final_string = "Ham"
            elif new_string == '1':
                final_string = "Spam"
            else:
                final_string = "Human checking is needed at this time for validation."
            self.new_test_dataframe.loc[self.i] = [message, len(message), final_string]
            self.i = self.i + 1

        # Show Evaluation information from prediction over the entire test set
        print('Accuracy Score: {}'.format(accuracy_score(y_test, self.predictions)))
        print('Precision Score: {}'.format(precision_score(y_test, self.predictions)))
        print('Recall Score: {}'.format(recall_score(y_test, self.predictions)))
        print('F1 Score: {}'.format(f1_score(y_test, self.predictions)))


        #Histogram of date
        # self.df.hist(column='Length', by='Label', bins=50, figsize=(10,4))
        # plt.show()

        # def plot_hist():
        #     fig_hist = self.df.hist(column='Length', by='Label', bins=50, figsize=(10,4))
        #     plt.show()
        #     return fig_hist


        # Make heatmap from Bag of words 'training_data'
        # conf_matrix = confusion_matrix(y_test, self.predictions)
        # fig, ax = plt.subplots(figsize=(8, 6))
        # df_cm = pd.DataFrame(conf_matrix, index=self.mnb.classes_,
        #                      columns=self.mnb.classes_)
        # sns.heatmap(df_cm, annot=True, fmt="d", ax=ax)
        # plt.ylabel('True Label')
        # plt.xlabel('Predicted Label')
        # plt.show()

        # Make WordClouds of each the training set and the test set
        # Create WordClouds for the Test and the Training set
        """ UNCOMMENT TO GET YOUR WORD CLOUDS BACK """
        # training_frames, test_frames = dataCleaning.make_sets(self.df, self.training_frames, self.test_frames)
        # wordCloudGen.show_wordcloud(training_frames, "Training Set")
        # wordCloudGen.show_wordcloud(test_frames, "Test Set")


    '''                  TEST ALL BELOW                    '''
    '''                UNCOMMENT ALL LINES BELOW                '''





