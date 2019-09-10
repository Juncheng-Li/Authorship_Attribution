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
from nltk import TweetTokenizer
from keras import utils
import tldextract
import joblib
import csv

tt = TweetTokenizer()


def clean_raw_data(sentence):
    tmp = tt.tokenize(sentence)
    # Convert url to domain
    if 'http://' in sentence:
        for i in range(len(tmp)):
            if 'http://' in tmp[i]:
                tmp[i] = tldextract.extract(tmp[i]).domain

    # Remove @handle
    if '@handle' in sentence:
        count = 0
        for i in range(len(tmp)):
            if tmp[i] == '@handle':
                count += 1
        for n in range(count):
            tmp.remove('@handle')

    # Remove stop words
    # stop_words = stopwords.words('english')
    # index = []
    # for i in range(len(tmp)):
    #     if tmp[i] in stop_words:
    #         index.append(tmp[i])
    # for element in index:
    #     tmp.remove(element)

    cleaned_sentence = " ".join(tmp)
    return cleaned_sentence

# Parameters
choose_model = 'svm'

# Read data
df = pd.read_csv(r'./whodunnit/train_tweets.txt', names=['label', 'sentence'], sep='\t', quoting=csv.QUOTE_NONE)
df_test = pd.read_csv(r'./whodunnit/test_tweets_unlabeled.txt', names=['sentence'], sep='\t',
                      quoting=csv.QUOTE_NONE)

# Clean data
df['sentence'] = df['sentence'].apply(clean_raw_data)

# Initialize vectorizer
# word_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1, max_features=15000)
# char_vectorizer = CountVectorizer(analyzer='char', stop_words='english', ngram_range=(2, 4), max_features=50000)
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_df=1, max_features=1)

# Prepare Training Data
sentences_train, sentences_test, y_train, y_test = train_test_split(df.sentence, df.label, test_size=0.1, random_state=42)
tfidf_vectorizer.fit(df.sentence)
x_train = tfidf_vectorizer.transform(sentences_train)
print(x_train.shape)
x_test = tfidf_vectorizer.transform(sentences_test)


if choose_model is 'RF':
    # Train and validate
    randomForest = RandomForestClassifier()
    randomForest.fit(x_train, y_train)
    validation_predictions = randomForest.predict(x_test)
    print(metrics.accuracy_score(y_test, validation_predictions))

    # Predictions
    df_test['sentence'] = df_test['sentence'].apply(clean_raw_data)
    print(df_test['sentence'].shape)
    test_sentences = tfidf_vectorizer.transform(df_test.sentence)
    predictions = randomForest.predict(test_sentences)
    print(predictions)
    # Save predictions
    df_predictions = pd.DataFrame(predictions, columns=['Predicted'])
    df_index = pd.DataFrame(list(range(1, len(predictions) + 1)), columns=['Id'])
    df_predictions = pd.concat([df_index, df_predictions], axis=1)
    df_predictions.to_csv(r'TFIDF_RF_predictions.csv', sep=',', index=False)

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
    df_test['sentence'] = df_test['sentence'].apply(clean_raw_data)
    print(df_test['sentence'].shape)
    test_sentences = tfidf_vectorizer.transform(df_test.sentence)
    predictions = svm.predict(test_sentences)
    print(predictions)
    # Save predictions
    df_predictions = pd.DataFrame(predictions, columns=['Predicted'])
    df_index = pd.DataFrame(list(range(1, len(predictions) + 1)), columns=['Id'])
    df_predictions = pd.concat([df_index, df_predictions], axis=1)
    df_predictions.to_csv(r'TFIDF_SVM_predictions.csv', sep=',', index=False)

    # save model
    model_filename = "Tfidf_svm.pkl"
    joblib.dump(svm, model_filename)


