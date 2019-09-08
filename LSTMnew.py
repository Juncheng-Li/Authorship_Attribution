import numpy as np
import pandas as pd
import csv
import nltk
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tldextract
nltk.download('stopwords')


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
    stop_words = stopwords.words('english')
    index = []
    for i in range(len(tmp)):
        if tmp[i] in stop_words:
            index.append(tmp[i])
    for element in index:
        tmp.remove(element)

    cleaned_sentence = " ".join(tmp)
    return cleaned_sentence


# Parameters
num_words = 50000
sequence_length = 250
epochs = 100
batch_size = 128

# Load data
df = pd.read_csv(r'./whodunnit/train_tweets.txt', names=['label', 'sentence'], sep='\t', quoting=csv.QUOTE_NONE)
df_test = pd.read_csv(r'./whodunnit/test_tweets_unlabeled.txt', names=['sentence'], sep='\t',
                      quoting=csv.QUOTE_NONE)

# Clean data
tt = TweetTokenizer()
print(df)
df['sentence'] = df['sentence'].apply(clean_raw_data)
print(df)

# Feature extraction
tokenizer = Tokenizer(num_words=num_words, filters='"#$%&()*+,-./:<=>?@\^_`|', lower=True)
tokenizer.fit_on_texts(df['sentence'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
x_train = tokenizer.texts_to_sequences(df['sentence'].values)
x_train = pad_sequences(x_train, maxlen=sequence_length)
print(x_train.shape)


# y_train = pd.get_dummies(df['label']).values
# print(y_train)
# print('label shape:', y_train.shape)

# Label encoding and convert on one-hot
y_train = df['label']
labelencoder = LabelEncoder()
labelencoder.fit(y_train)
num_classes = len(labelencoder.classes_)
print("num_classes:  " + str(num_classes))
y_train = labelencoder.transform(y_train)
y_train = to_categorical(y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Construct LSTM and start training
model = Sequential()
model.add(Embedding(num_words, 100, input_length=sequence_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# Evaluation
model.save(r'keras-lstm.h5')

# Prepare predict data
df_test['sentence'] = df_test['sentence'].apply(clean_raw_data)
print(df_test['sentence'].shape)
x_submit = tokenizer.texts_to_sequences(df_test['sentence'].values)
x_submit = pad_sequences(x_submit, maxlen=sequence_length)

# Predict
print("Predicting unlabelled data...")
predictions = model.predict(x_submit)
print(predictions)
predictions = np.argmax(predictions, axis=1)
print(predictions)
predictions = labelencoder.inverse_transform(predictions)
print(predictions)

# Save predictions
df_predictions = pd.DataFrame(predictions, columns=['Predicted'])
df_index = pd.DataFrame(list(range(1, len(predictions)+1)), columns=['Id'])
df_predictions = pd.concat([df_index, df_predictions], axis=1)
df_predictions.to_csv(r'LSTM_predictions.csv', sep=',', index=False)
