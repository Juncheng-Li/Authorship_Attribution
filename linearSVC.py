import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import csv
import tldextract
from nltk import TweetTokenizer

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

# Load data
df = pd.read_csv(r'./whodunnit/train_tweets.txt', names=['label', 'sentence'], sep='\t', quoting=csv.QUOTE_NONE)
df_test = pd.read_csv(r'./whodunnit/test_tweets_unlabeled.txt', names=['sentence'], sep='\t',
                      quoting=csv.QUOTE_NONE)

# Clean data
tt = TweetTokenizer()
print(df)
df['sentence'] = df['sentence'].apply(clean_raw_data)
print(df)
x_train = df['sentence']
y_train = df['label']
# x_train, x_test, y_train, y_test = train_test_split(df['sentence'], df['label'], test_size=0.01, random_state=42)

# Feature extraction
tfidf_vectorizer1 = TfidfVectorizer(analyzer='word', stop_words=None, ngram_range=(1, 3), min_df=1, max_features=100000)
tfidf_vectorizer2 = TfidfVectorizer(analyzer='char', stop_words=None, ngram_range=(1, 4), max_features=50000)

tfidf_vectorizer1.fit(df['sentence'].values)
tfidf_vectorizer2.fit(df['sentence'].values)
vec1 = tfidf_vectorizer1.transform(x_train)
vec2 = tfidf_vectorizer2.transform(x_train)
x_train = hstack([vec1, vec2])

# Conctruct model and train
svm = LinearSVC(verbose=True)
print("Start training...")
svm.fit(x_train, y_train)
# y_pred_test = SVM.predict(x_test)

# Prepare predicion data
print(df_test['sentence'].shape)
vec1 = tfidf_vectorizer1.transform(df_test['sentence'])
vec2 = tfidf_vectorizer2.transform(df_test['sentence'])
x_submit = hstack([vec1, vec2])
predictions = svm.predict(x_submit)

# Save predictions
df_predictions = pd.DataFrame(predictions, columns=['Predicted'])
df_index = pd.DataFrame(list(range(1, len(predictions)+1)), columns=['Id'])
df_predictions = pd.concat([df_index, df_predictions], axis=1)
df_predictions.to_csv(r'svm_resutls.csv', sep=',', index=False)

# Save model
joblib.dump(svm, 'svc_tfidf.pkl')
