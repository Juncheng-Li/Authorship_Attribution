import pandas as pd
import csv
import nltk
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer
import tldextract
nltk.download('stopwords')


def clean_raw_data(sentence):
    cleaned = tt.tokenize(sentence)
    print(cleaned)
    # Convert url to domain
    if 'http' in cleaned:
        for i in range(len(cleaned)):
            if 'http' in cleaned[i]:
                cleaned[i] = tldextract.extract(cleaned[i]).domain

    # Remove @handle
    if '@handle' in cleaned:
        count = 0
        for i in range(len(cleaned)):
            if cleaned[i] == '@handle':
                count += 1
        for n in range(count):
            cleaned.remove('@handle')

    # Remove stop words
    stop_words = stopwords.words('english')
    index = []
    for i in range(len(cleaned)):
        if cleaned[i] in stop_words:
            index.append(cleaned[i])
    for element in index:
        cleaned.remove(element)

    return cleaned


df = pd.read_csv(r'./whodunnit/train_tweets.txt', names=['label', 'sentence'], sep='\t', quoting=csv.QUOTE_NONE)
df_test = pd.read_csv(r'./whodunnit/test_tweets_unlabeled.txt', names=['label', 'sentence'], sep='\t', quoting=csv.QUOTE_NONE)

tt = TweetTokenizer()

cleaned = clean_raw_data('here is the @handle')
print(cleaned)
