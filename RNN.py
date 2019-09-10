import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.sparse import hstack
from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical
from nltk import TweetTokenizer
import tldextract
import matplotlib.pyplot as plt


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

# Hyper-Parameters
epoch = 30
batch_size = 64
feature_size = 100000

# Read files
df = pd.read_csv(r'./whodunnit/train_tweets.txt', names=['label', 'sentence'], sep='\t', quoting=csv.QUOTE_NONE)
df_test = pd.read_csv(r'./whodunnit/test_tweets_unlabeled.txt', names=['sentence'], sep='\t',
                      quoting=csv.QUOTE_NONE)
print(df.shape)
assert df.shape != 328932, print("Shape Error!!!!!")

# Clean data
tt = TweetTokenizer()
print(df)
df['sentence'] = df['sentence'].apply(clean_raw_data)
print(df)

# Initialise vectoriser
tf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=feature_size)
# Preparing train and test dataset
sentences_train = df.sentence
tf_vectorizer.fit(sentences_train)
x_train = tf_vectorizer.transform(sentences_train)

# Label encoding and convert on one-hot
y_train = df['label']
labelencoder = LabelEncoder()
labelencoder.fit(y_train)
num_classes = len(labelencoder.classes_)
print("num_classes:  " + str(num_classes))
y_train = labelencoder.transform(y_train)
y_train = to_categorical(y_train)

# Model
model = Sequential()
model.add(layers.Dense(16, input_shape=(feature_size,), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=epoch,
                    verbose=2,
                    shuffle=True,
                    validation_split=0.1,
                    batch_size=batch_size)

# Save model
model.save(r'RNN.h5')

# Prepare predict data
df_test['sentence'] = df_test['sentence'].apply(clean_raw_data)
print(df_test['sentence'].shape)
x_submit = tf_vectorizer.transform(df_test['sentence'])

# Predict
print("Predicting unlabelled data...")
predictions = model.predict(x_submit)
predictions = np.argmax(predictions, axis=1)
predictions = labelencoder.inverse_transform(predictions)

# Save predictions
df_predictions = pd.DataFrame(predictions, columns=['Predicted'])
df_index = pd.DataFrame(list(range(1, len(predictions)+1)), columns=['Id'])
df_predictions = pd.concat([df_index, df_predictions], axis=1)
df_predictions.to_csv(r'RNN_predictions.csv', sep=',', index=False)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()