import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wordcloud
import nltk
import warnings
import seaborn as sns
import scipy as sp

# Variables that will be used inside of the class
dfHam = None
dfSpam = None


# Class constructor
def __init__(self, dataFrame):
    self.dataFrame = dataFrame


def show_wordcloud(df, title):
    text = ' '.join(df['SMS'].astype(str).tolist())
    stopwords = set(wordcloud.STOPWORDS)
    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords, background_color="#ffa78c",
                                        width=3000, height=2000).generate(text)
    #plt.figure(figsize=(15,15), frameon=True)
    plt.imshow(fig_wordcloud)
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()

def show_wordcloud_alt(df, title, ax):
    text = ' '.join(df['SMS'].astype(str).tolist())
    stopwords = set(wordcloud.STOPWORDS)
    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords, background_color="#ffa78c",
                                        width=3000, height=2000).generate(text)
    plt.figure(figsize=(15,15), frameon=True)
    plt.imshow(fig_wordcloud)
    plt.axis(ax)
    plt.title(title, fontsize=20)
    plt.show()