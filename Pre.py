import pandas as pd
import numpy as np
import csv
import time
import nltk
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
import tldextract
nltk.download('stopwords')


def clean_raw_data(sentence):
    tmp = tt.tokenize(sentence)
    # Convert url to domain
    if 'http' in tmp:
        for i in range(len(tmp)):
            if 'http' in tmp[i]:
                tmp[i] = tldextract.extract(tmp[i]).domain

    # Remove @handle
    if '@handle' in tmp:
        count = 0
        for i in range(len(tmp)):
            if tmp[i] == '@handle':
                count += 1
        for n in range(count):
            tmp.remove('@handle')

    # Remove stop words
    stop_words = stopwords.words('english')
    index = []
    for i in range(len(tmp)):
        if tmp[i] in stop_words:
            index.append(tmp[i])
    for element in index:
        tmp.remove(element)

    cleaned_sentence = " ".join(tmp)
    return cleaned_sentence


df = pd.read_csv(r'./whodunnit/train_tweets.txt', names=['label', 'sentence'], sep='\t', quoting=csv.QUOTE_NONE)
df_test = pd.read_csv(r'./whodunnit/test_tweets_unlabeled.txt', names=['label', 'sentence'], sep='\t', quoting=csv.QUOTE_NONE)
print(df.sentence.shape[0])
tt = TweetTokenizer()

# tic = time.clock()
# print(df)
# df['sentence'] = df['sentence'].apply(clean_raw_data)
# print(df)
# toc = time.clock()
# print("time processed: ")
# print(toc - tic)

# Label processing
y_train = df['label']
print(y_train)
labelencoder = LabelEncoder()
labelencoder.fit(y_train)
num_classes = len(labelencoder.classes_)
print("num_classes:  " + str(num_classes))
y_train = labelencoder.transform(y_train)
print(y_train)
y_train = to_categorical(y_train)
print(y_train)

# Reverse Label encoding
y_train = np.argmax(y_train, axis=1)
print(y_train)
y_train = labelencoder.inverse_transform(y_train)
print(y_train)
