from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import *
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import pandas as pd


class Data:
    
    # Variables
    column_names = ["label", "sms"]
    training_set = pd.DataFrame(columns=column_names)
    test_set = pd.DataFrame(columns=column_names)
    training_frames = pd.DataFrame(columns=column_names)
    test_frames = pd.DataFrame(columns=column_names)
    new_test_dataframe = pd.DataFrame(columns=['SMS', 'Length', 'Result'])
    i = 0

    # MAIN BODY OF APPLICATION
    
    def __init__(self):
        
        # Variables:
        self.column_names = ["label", "sms"]
        self.training_set = pd.DataFrame(columns=self.column_names)
        self.test_set = pd.DataFrame(columns=self.column_names)
        self.training_frames = pd.DataFrame(columns=self.column_names)
        self.test_frames = pd.DataFrame(columns=self.column_names)
        self.new_test_dataframe = pd.DataFrame(columns=['SMS', 'Length', 'Result'])
        self.i = 0
        self.mnb = MultinomialNB()
        self.lr = LogisticRegression()
        self.predictions = None
        self.lr_predictions = None
        self.df = None
        self.df2 = None
        self.df3 = None
        self.count_vector = None
        self.accuracy_score = None
        self.precision_score = None
        self.recall_score = None
        self.f1_score = None
        self.lr_accuracy_score = None
        self.lr_precision_score = None
        self.lr_recall_score = None
        self.lr_f1_score = None

        # Import the data from the dataset
        # file_path = '/home/televator/Coding/spam_sample/smsspamcollection/SMSSpamCollection'
        file_path = 'SMSSpamCollection.csv'
        
        # Take data from CSV dataset and put it into a Pandas DataFrame
        # First dataset to be used with Multinomial Naieve Bayes
        self.df = pd.read_csv(file_path, sep='\t', header=None, names=['Label', 'SMS'])
        self.df['Length'] = self.df['SMS'].apply(len)
        
        # Second dataset to be used with GUI
        self.df2 = pd.read_csv(file_path, sep='\t', header=None, names=['Label', 'SMS'])
        self.df2['Length'] = self.df2['SMS'].apply(len)
        
        # Third dataset to be used with
        """ POSSIBLY USE A THIRD DATASET """
        self.df3 = pd.read_csv(file_path, sep='\t', header=None, names=['Label', 'SMS'])
        self.df3['Length'] = self.df3['SMS'].apply(len)
        
        # Create keys for the labels
        self.df.loc[:, 'Label'] = self.df.Label.map({'ham':0, 'spam':1})
        
        # Split the data, and initiate the count vectorizer
        x_train, x_test, y_train, y_test = train_test_split(self.df['SMS'], self.df['Label'], test_size=0.20, random_state=1)
        
        # Create the count vectorizer
        self.count_vector = CountVectorizer()
        
        # Vectorize the data, then test it in the 'mnb' algorithm
        training_data = self.count_vector.fit_transform(x_train)
        testing_data = self.count_vector.transform(x_test)
        
        # Train the Multinomial Naive Bayes model
        self.mnb = MultinomialNB()
        self.mnb.fit(training_data, y_train)
        self.predictions = self.mnb.predict(testing_data)
        
        # Train the Logisitic regression model
        self.lr = LogisticRegression(solver='liblinear', penalty='l1')
        self.lr.fit(training_data, y_train)
        self.lr_predictions = self.lr.predict(testing_data)
        
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

        # Show Evaluation information from prediction over the entire test set of MNB
        self.accuracy_score = str(accuracy_score(y_test, self.predictions))
        self.precision_score = str(precision_score(y_test, self.predictions))
        self.recall_score = str(recall_score(y_test, self.predictions))
        self. f1_score = str(f1_score(y_test, self.predictions))
        
        # Get the same evaulation data from the logisitic regression model
        self.lr_accuracy_score = str(accuracy_score(y_test, self.lr_predictions))
        self.lr_precision_score = str(precision_score(y_test, self.lr_predictions))
        self.lr_recall_score = str(recall_score(y_test, self.lr_predictions))
        self.lr_f1_score = str(f1_score(y_test, self.lr_predictions))


