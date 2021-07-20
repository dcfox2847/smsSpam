import pandas as pd

class dataCleaning:

    def __init__(self, name=None):
        self.name = name

    def first_step(name):
        print("This is from the first step function. And your name is " + name + ".")

    def add_length_field(df):
        df['length'] = df['sms'].apply(len)

    def get_ham(df):
        return df[df['label'] == "ham"].copy()

    def get_spam(df):
        return df[df['label'] == "spam"].copy()

    # Begin the cleaning of data

    def make_sets(df, training_set, test_set):
        # Randomize the dataset
        data_randomized = df.sample(frac=1, random_state = 1)
        # Calculate the index for the split
        training_test_index = round(len(data_randomized) * 0.8)
        #Split the data into the training and test sets
        training_set = data_randomized[:training_test_index].reset_index(drop=True)
        test_set = data_randomized[training_test_index:].reset_index(drop=True)
        return training_set, test_set

    def make_lower(df):
        # Remove all punctuations
        df['sms'] = df['sms'].str.replace('\W', ' ')
        df['sms'] = df['sms'].str.lower()
        return df

    def make_vocab(df):
        df['sms'] = df['sms'].str.split()
        # Make a list variable to hold the separate words from the sms messages
        vocabulary = []
        for sms in df['sms']:
            for word in sms:
                vocabulary.append(word)
        # Remove duplicates using the 'set' function
        vocabulary = list(set(vocabulary))
        return vocabulary

    def finalize_set(df, training_vocab):
        word_counts_per_sms = {unique_word: [0] * len(df['sms']) for unique_word in training_vocab}
        for index, sms in enumerate(df['sms']):
            for word in sms:
                word_counts_per_sms[word][index] += 1
        word_counts = pd.DataFrame(word_counts_per_sms)
        training_set_clean = pd.concat([df, word_counts], axis=1)
        return training_set_clean

