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


# Variables
column_names = ["label", "sms"]
training_set = pd.DataFrame(columns=column_names)
test_set = pd.DataFrame(columns=column_names)
training_frames = pd.DataFrame(columns=column_names)
test_frames = pd.DataFrame(columns=column_names)
new_test_dataframe = pd.DataFrame(columns=['SMS', 'Length', 'Result'])
i = 0


# MAIN BODY OF APPLICATION
# Import the data from the dataset
file_path = '/home/televator/Coding/spam_sample/smsspamcollection/SMSSpamCollection'
# Take data from CSV dataset and put it into a Pandas DataFrame
df = pd.read_csv(file_path, sep='\t', header=None, names=['Label', 'SMS'])
df['Length'] = df['SMS'].apply(len)
df.loc[:, 'Label'] = df.Label.map({'ham':0, 'spam':1})
# Split the data, and initiate the count vectorizer
x_train, x_test, y_train, y_test = train_test_split(df['SMS'], df['Label'], test_size=0.20, random_state=1)
# Create the count vectorizer
count_vector = CountVectorizer()
# Vectorize the data, then test it in the 'mnb' algorithm
training_data = count_vector.fit_transform(x_train)
testing_data = count_vector.transform(x_test)
mnb = MultinomialNB()
mnb.fit(training_data, y_train)
predictions = mnb.predict(testing_data)
# Create a new dataframe that will hold all result data for plotting
# This data frame has the SMS message, the length of the message, and will show 'Spam' or 'Ham' as a result
for row in x_test:
    message = row
    text = count_vector.transform([message])
    prediction = mnb.predict(text)
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
    new_test_dataframe.loc[i] = [message, len(message), final_string]
    i = i + 1

# Show Evaluation information from prediction over the entire test set
print('Accuracy Score: {}'.format(accuracy_score(y_test, predictions)))
print('Precision Score: {}'.format(precision_score(y_test, predictions)))
print('Recall Score: {}'.format(recall_score(y_test, predictions)))
print('F1 Score: {}'.format(f1_score(y_test, predictions)))


# Histogram showing the lengths of each message by the label 'spam' or 'ham'
df.hist(column='Length', by='Label', bins=50, figsize=(10,4))
plt.show()

# Make heatmap from Bag of words 'training_data'
conf_matrix = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(8, 6))
df_cm = pd.DataFrame(conf_matrix, index=mnb.classes_,
                     columns=mnb.classes_)
sns.heatmap(df_cm, annot=True, fmt="d", ax=ax)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Make WordClouds of each the training set and the test set
# Create WordClouds for the Test and the Training set
training_frames, test_frames = dataCleaning.make_sets(df, training_frames, test_frames)
wordCloudGen.show_wordcloud(training_frames, "Training Set")
wordCloudGen.show_wordcloud(test_frames, "Test Set")

'''                  TEST ALL BELOW                    '''
'''                UNCOMMENT ALL LINES BELOW                '''

# # Create a 'bag of words'
# countvec = CountVectorizer(ngram_range=(1,4), stop_words='english', strip_accents='unicode', max_features=1000)
# # SPLIT THE SETS INTO TRAINING AND TESTING HERE
# training_set, test_set = dataCleaning.make_sets(df, training_set, test_set)
# # Create bag of words for both the training and test set
# training_bow = countvec.fit_transform(training_set.SMS)
# # Prepare training data
# train_X_train = training_bow.toarray()
# train_Y_train = training_set.Label.values
# # Instantiate the classifier, and then train it
# mnb = MultinomialNB()
# mnb.fit(train_X_train, train_Y_train)
#
# # TAKE THE TEST SET DATA FRAME, AND ADD ANOTHER COLUMN FOR THE RESULTS
# test_set['Result'] = ''
# # Iterate over dataframe
# for index, row in test_set.iterrows():
#     message = row['SMS']
#     cleaned_message = countvec.transform([message])
#     res = mnb.predict(cleaned_message)
#     final_res = np.array_str(res)
#     characters_to_remove = "',[,]"
#     new_string = final_res
#     for character in characters_to_remove:
#         new_string = new_string.replace(character, '')
#     row['Result'] = new_string
#
# print('Accuracy Score: {}'.format(accuracy_score()))
# # Create bar chart showing totals of spam and ham
# test_set.Label.value_counts().plot.bar()
# plt.show()
# # Create WordClouds for the Test and the Training set
# wordCloudGen.show_wordcloud(training_set, "Training Set")
# wordCloudGen.show_wordcloud(test_set, "Test Set")
# # TEST SAMPLE
# # text = countvec.transform(['You could be selected for a cruise! TXT now!'])
# # print(mnb.predict(text))



