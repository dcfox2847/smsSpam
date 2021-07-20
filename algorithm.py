import wordCloudGen
from dataCleaning import *

import re
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px
import sklearn
import wordcloud
import nltk
import warnings
import seaborn as sns
import scipy as sp

# Global Variables
column_names = ["label", "sms"]
training_set = pd.DataFrame(columns=column_names)
test_set = pd.DataFrame(columns=column_names)
# training_set = pd.DataFrame(columns=column_names)
# test_set = pd.DataFrame(columns=column_names)

# Constructor

# Begin the parsing and utilization of data from dataset

# def main_algorithm(self, training_set, test_set, column_names):
# Import the data from the dataset
file_path = '/home/televator/Coding/spam_sample/smsspamcollection/SMSSpamCollection'
# Take data from CSV dataset and put it into a Pandas DataFrame
df = pd.read_csv(file_path, sep='\t', header=None, names=column_names)
# Use the 'make_sets" function from the data cleaning class
training_set, test_set = dataCleaning.make_sets(df, training_set, test_set)
# Remove punctuation and make all lower case
training_set= dataCleaning.make_lower(training_set)
test_set = dataCleaning.make_lower(test_set)
# Create a vocabulary list
training_vocab = dataCleaning.make_vocab(training_set)
test_vocab = dataCleaning.make_vocab(test_set)
# Create a dictionary of data for the training set
final_training = dataCleaning.finalize_set(training_set, training_vocab)
# Isolate the Spam from the Ham
spam_messages = final_training[final_training['label'] == 'spam']
ham_messages = final_training[final_training['label'] == 'ham']
# P(spam) and P(ham)
p_spam = len(spam_messages) / len(final_training)
p_ham = len(ham_messages) / len(final_training)
# N_Spam and N_ham
n_words_per_spam_message = spam_messages['sms'].apply(len)
n_spam = n_words_per_spam_message.sum()
print("n_spam = " + str(n_spam))
n_words_per_ham_message = ham_messages['sms'].apply(len)
n_ham = n_words_per_ham_message.sum()
print("n_ham = " + str(n_ham))
# Calculate N_vocabulary
n_vocabulary = len(training_vocab)
print("n_vocabulary = " + str(n_vocabulary))
# Set value for laplace smoothing
alpha = 1
# Initiate Parameters
parameters_spam = {unique_word: [0] for unique_word in training_vocab}
parameters_ham = {unique_word: [0] for unique_word in training_vocab}
# Perform calculation
for word in training_vocab:
    n_word_given_spam = spam_messages[word].sum() + alpha
    denom = n_spam + (n_vocabulary * alpha)
    p_word_given_spam = n_word_given_spam / denom
    parameters_spam[word] = p_word_given_spam

    # n_word_given_ham = ham_messages[word].sum()
    # p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
    # parameters_ham[word] = p_word_given_ham

# def classify(message):
#     message = re.sub('\W', ' ', message)
#     message = message.lower().split()
#     p_spam_given_message = p_spam
#     p_ham_given_message = p_ham
#
#     for word in message:
#         if word in parameters_spam:
#             p_spam_given_message *= parameters_spam[word]
#
#         if word in parameters_ham:
#             p_ham_given_message *= parameters_ham[word]
#
#     print('P(Spam|message):', p_spam_given_message)
#     print('P(Ham|message):', p_ham_given_message)
#
#     if p_ham_given_message > p_spam_given_message:
#         print('Label; ham')
#     elif p_ham_given_message < p_spam_given_message:
#         print('Label: spam')
#     else:
#         print('Equal probabilities, have a human classify this!')
#
#
# # TESTING THIS TO BE USED AS THE CLASS CONSTRUCTOR, AND FOR CALLING AND PARSING ALL DATA WHILE THE
# classify('WINNER! This is the secret code to unlock the money: C3421.')







