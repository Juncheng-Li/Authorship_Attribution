import pandas as pd
import numpy as np
import scipy as sp
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras import utils
import joblib
import csv

choose_model = 'svm'

df = pd.read_csv(r'./whodunnit/train_tweets.txt', names=['label', 'sentence'], sep='\t', quoting=csv.QUOTE_NONE)
print(df)

# Initialize vectorizer
# word_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1, max_features=15000)
# char_vectorizer = CountVectorizer(analyzer='char', stop_words='english', ngram_range=(2, 4), max_features=50000)
tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 2), max_df=1, max_features=1)

# Prepare Training Data
sentences_train, sentences_test, y_train, y_test = train_test_split(df.sentence, df.label, test_size=0.3, random_state=42)
tfidf_vectorizer.fit(df.sentence)
x_train = tfidf_vectorizer.transform(sentences_train)
print(x_train.shape)
x_test = tfidf_vectorizer.transform(sentences_test)

# # label encoder
# labelencoder = LabelEncoder()
# labelencoder.fit(df.label)
# print("num_classes: ")
# num_classes = len(labelencoder.classes_)
# print(num_classes)
# y_train = labelencoder.transform(y_train)
# y_test = labelencoder.transform(y_test)


if choose_model is 'RF':
    # Train and validate
    randomForest = RandomForestClassifier()
    randomForest.fit(x_train, y_train)
    validation_predictions = randomForest.predict(x_test)
    print(metrics.accuracy_score(y_test, validation_predictions))

    # Predictions
    df_test = pd.read_csv(r'./whodunnit/test_tweets_unlabeled', names=['sentence'], sep='\t', quoting=csv.QUOTE_NONE)
    test_sentences = tfidf_vectorizer.transform(df.sentence)
    predictions = randomForest.predict(test_sentences)
    predictions = pd.DataFrame(predictions)
    predictions = pd.concat([df.sentence, predictions])
    predictions.to_csv(r'./Rf_predictions.csv')

    # save model
    model_filename = "Tfidf_randomForest.pkl"
    joblib.dump(randomForest, model_filename)

if choose_model is 'svm':
    # Train and validate
    svm = LinearSVC()
    svm.fit(x_train, y_train)
    validation_predictions = svm.predict(x_test)
    print(metrics.accuracy_score(y_test, validation_predictions))

    # Predictions
    df_test = pd.read_csv(r'./whodunnit/test_tweets_unlabeled', names=['sentence'], sep='\t', quoting=csv.QUOTE_NONE)
    test_sentences = tfidf_vectorizer.transform(df.sentence)
    predictions = svm.predict(test_sentences)
    predictions = pd.DataFrame(predictions)
    predictions = pd.concat([df.sentence, predictions])
    predictions.to_csv(r'./Rf_predictions.csv')

    # save model
    model_filename = "Tfidf_randomForest.pkl"
    joblib.dump(svm, model_filename)


