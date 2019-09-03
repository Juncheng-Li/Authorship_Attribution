import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.sparse import hstack
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt

# Hyper-Parameters
filePath = './whodunnit/train_tweets.txt'
num_class = 9297
choose_model = 'RNN'    #choose from 'LR', 'RNN'
epoch = 10
word_dim = 2100


# Read files
df = pd.read_csv(filePath, names=['label', 'sentence'], sep='\t', quoting=csv.QUOTE_NONE)
print(df.shape)
assert df.shape != 328932, print("Shape Error!!!!!")

# Vectorising most frequent words
vectorizer_1 = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1, max_features=15000)
vectorizer_2 = CountVectorizer(analyzer='char', stop_words='english', ngram_range=(2, 4), max_features=50000)

# Preparing train and test dataset
sentences_train, sentences_test, y_train, y_test = train_test_split(df.sentence, df.label, test_size=0.25, random_state=42)
vectorizer_1.fit(sentences_train)
vectorizer_2.fit(sentences_train)
x_train_1 = vectorizer_1.transform(sentences_train)
x_train_2 = vectorizer_2.transform(sentences_train)
x_train = hstack([x_train_1, x_train_2])
vectorizer_1.fit(sentences_test)
vectorizer_2.fit(sentences_test)
x_test_1 = vectorizer_1.transform(sentences_test)
x_test_2 = vectorizer_2.transform(sentences_test)
x_test = hstack([x_test_1, x_test_2])
print("X_train size: " + str(x_train.shape()) + " X_test size: " + str(x_test.shape()))

labelencoder = LabelEncoder()
labelencoder.fit(y_train)
print("num_classes: ")
print(len(labelencoder.classes_))
onehotencoder = OneHotEncoder(categorical_features=[0])
y_train = onehotencoder.fit_transform(y_train).toarray()
y_test = onehotencoder.fit_transform(y_test).toarray()
print("y_test: ")
print(y_test)

# Model - Logistic Regression
if choose_model is 'LR':
    model = LogisticRegression()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print("Logistic Regression - Accuracy: ", score)

# Model - Basic RNN
if choose_model is 'RNN':
    model = Sequential()
    model.add(layers.Dense(10, input_dim=word_dim, activation='relu'))
    model.add(layers.Dense(num_class, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train,
                        epochs=epoch,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        batch_size=16)

    loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plt.style.use('ggplot')
    def plot_history(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
    plot_history(history)

