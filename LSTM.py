import itertools
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize

STOPWORDS = set(stopwords.words('english'))
from bs4 import BeautifulSoup
from IPython.core.interactiveshell import InteractiveShell
import plotly.figure_factory as ff

file = "./whodunnit/train_tweets.txt"
temp = []
with open(file, 'r', encoding='utf-8') as data:
    for line in data:
        row = []
        line = line.replace('\t', " ")
        elem = line.strip().split(" ")
        row.append(elem[0])
        row.append(" ".join(elem[1:]))
        temp.append(row)

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer


def text_process(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text_processed = tokenizer.tokenize(text)
    text_processed = ' '.join(word for word in text_processed if word not in STOPWORDS)
    #     porter_stemmer = PorterStemmer()
    #     text_processed = [porter_stemmer.stem(word) for word in text_processed]
    return text_processed


def clean_df(tw):
    tw["Tweet"].replace(r'http.?://[^\s]+[\s]?', '', regex=True, inplace=True)
    tw['Tweet'] = tw['Tweet'].str.lower()
    tw["Tweet"].replace(r"@\S+", " ", regex=True, inplace=True)
    #     tw["Tweet"].replace(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})+", "DATE", regex=True,inplace=True)
    #     tw["Tweet"].replace(r"(\d{1,2})[/:](\d{2})[/:](\d{2})?(am|pm)+", "TIME", regex=True,inplace=True)
    #     tw["Tweet"].replace(r"(\d{1,2})[/:](\d{2})?(am|pm)+", "TIME", regex=True,inplace=True)
    #     tw["Tweet"].replace(r"\d+", "NUM", regex=True,inplace=True)
    tw["Tweet"].replace('[^a-zA-Z\s]', '', regex=True, inplace=True)
    tw['num_of_words'] = tw["Tweet"].str.split().apply(len)
    #     tw.drop(tw[tw.num_of_words<4].index, inplace=True)
    return tw


df = pd.DataFrame(temp, columns=["User", "Tweet"])
df = clean_df(df)
df['Tweet'] = df['Tweet'].apply(text_process)

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Tweet'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# WOrd Embedding and pad sequence
X = tokenizer.texts_to_sequences(df['Tweet'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['User']).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.5))
model.add(Dense(9297, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

epochs = 90
batch_size = 128

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.15,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
model.save(r'keras-lstm.h5')
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
