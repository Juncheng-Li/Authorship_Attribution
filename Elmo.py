import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalAveragePooling1D, Dropout, Lambda
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import csv

def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

def build_model():
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    dense = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(embedding)
    pred = Dense(len(list_classes), activation='softmax')(dense)
    model = Model(inputs=[input_text], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# HyperParameters
max_features = 20000
maxlen = 100

# Preprocess
df = pd.read_csv('./whodunnit/train_tweets.txt', names=['label', 'sentence'], sep='\t', quoting=csv.QUOTE_NONE)
sentences_train, sentences_test, y_train, y_test = train_test_split(df.sentence, df.label, test_size=0.22, random_state=42)

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
embeddings = elmo(["the cat is on the mat", "what are you doing in evening"], signature="default", as_dict=True)["elmo"]

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer])
    message_embeddings = session.run(embeddings)

tokens_input = [["the", "cat", "is", "on", "the", "mat"], ["what", "are", "you", "doing", "in", "evening"]]
tokens_length = [6, 5]
embeddings = elmo(inputs={"tokens": tokens_input, "sequence_len": tokens_length}, signature="tokens", as_dict=True)["elmo"]
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = session.run(embeddings)



model_elmo = build_model()

model_elmo.summary()

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    history = model_elmo.fit(list_sentences_train, y, epochs=5, batch_size=2, validation_split=0.2)
    model_elmo.save_weights('./model_elmo_weights.h5')