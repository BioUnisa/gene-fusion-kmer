import argparse
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from keras import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from utils.file import create_training_path_file, create_conf_path_file

"""
def encode2train(input_file: str, code_function): 
    df = pd.read_csv(input_file, header=0, engine="python")
    df_seq = df.iloc[:, 1]
    X = np.array([code_function(x) for x in df_seq])
    y = np.array(df.iloc[:, 0])
    classes = np.unique(y)

    return X, y, classes


def reshape_data(X):
    n_samp, len_seq, wid_seq = X.shape
    if image_data_format() == "channels_first":
        X = X.reshape(n_samp, 1, len_seq, wid_seq)
    else:
        X = X.reshape(n_samp, len_seq, wid_seq, 1)

    return X


def create_model(_n_classes):
    _model = tf.keras.models.Sequential()

    _model.add(Conv2D(128, (3, 2), activation='relu', input_shape=X_train.shape[1:]))
    _model.add(Conv2D(64, (3, 2), activation='relu', input_shape=X_train.shape[1:]))
    _model.add(Conv2D(32, (2, 1), activation='relu', input_shape=X_train.shape[1:]))
    _model.add(Flatten())
    _model.add(Dense(128, activation='relu'))
    _model.add(Dense(_n_classes, activation='softmax'))

    return _model
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='input_file', action='store',
                        default='kmer-dataset', help='name of input file')

    args = parser.parse_args()

    # Check if input file exists
    conf_path_file = create_conf_path_file(args.input_file)
    training_path_file = create_training_path_file(args.input_file)
    if not os.path.isfile(conf_path_file) or not os.path.isfile(training_path_file):
        print("No such file.")
        print("Please run pre-processing.py first")

    # Load conf object
    conf = {}
    with open(conf_path_file, "rb") as conf_file:
        conf = pickle.load(conf_file)

    # Get X and y of training
    df = pd.read_csv(training_path_file, header=0, engine="python")
    id_genes = np.array(df.iloc[:, 0])
    kmers = np.array(df.iloc[:, 1])

    # Encode label train
    classes = conf["classes"]
    n_classes = conf["n_classes"]
    label_encoder = conf["label-encoder"]
    id_genes_encoded = label_encoder.transform(id_genes)
    id_genes_encoded = to_categorical(id_genes_encoded, num_classes=n_classes)

    # Tokenizing kmers
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(kmers)
    encoded_kmers = tokenizer.texts_to_sequences(kmers)
    max_length = max([len(s.split()) for s in kmers])
    kmers = pad_sequences(encoded_kmers, maxlen=max_length, padding='post')

    # Split dataset in training and validation set
    kmers_train, kmers_val, id_genes_train, id_genes_val = train_test_split(
        kmers, id_genes_encoded, test_size=0.25, random_state=42)

    vocab_size = len(tokenizer.word_index) + 1

    model = Sequential()
    model.add(Embedding(vocab_size, 20))
    model.add(Bidirectional(LSTM(20)))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)
    ]

    model.fit(kmers_train,
              id_genes_train,
              validation_data=(kmers_val, id_genes_val),
              batch_size=128,
              epochs=200,
              callbacks=callbacks)

    model.save('./training/gene-fusion-model.h5', save_format='h5')
