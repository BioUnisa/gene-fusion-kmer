import argparse
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from keras.backend import image_data_format
from keras.layers import Conv2D, Flatten, Dense
from keras.utils import to_categorical
from training.encoding import one_hot_encode as ohe
from utils.file import create_training_path_file, create_test_path_file, create_conf_path_file


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

    # Load if exists test file
    test_path_file = ""
    if conf["split"]:
        test_path_file = create_test_path_file(args.input_file)
        if not os.path.isfile(test_path_file):
            print("No such file.")
            print("Please run pre-processing.py first")

    # Get X, y and classes of training
    X_train, y_train, classes_train = encode2train(training_path_file, ohe)

    # Encode label train
    classes = conf["classes"]
    n_classes = conf["n_classes"]
    label_encoder = conf["label-encoder"]
    y_train_encoded = label_encoder.transform(y_train)
    y_train_encoded = to_categorical(y_train_encoded, num_classes=n_classes)

    X_train = reshape_data(X_train)

    print("Dimension of data: " + str(X_train.shape))
    print(f"Num of training classes: {len(classes_train)}")

    """
    model = create_model(n_classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    model.fit(X_train, y_train_encoded, batch_size=128, epochs=2)

    model.save('./training/gene-fusion-model.h5', save_format='h5')
    """

    model = tf.keras.models.load_model('./training/gene-fusion-model.h5')

    # Test if split is True
    if conf["split"]:
        # Load csv and get X and y
        df = pd.read_csv(test_path_file, header=0, engine="python")
        df_seq = df.iloc[:, 1]
        y = np.array(df.iloc[:, 0])

        # For each element
        for i in range(len(y)):
            print(f"Label to predict: {y[i]}")
            # Get kmer
            sequence = df_seq[i]
            n_kmers = (len(sequence) - conf["k-size"] + 1)
            kmers = [sequence[j:j + conf["k-size"]] for j in range(n_kmers)]
            X_test = np.array([ohe(kmer) for kmer in kmers])
            X_test = reshape_data(X_test)

            # Create y_train encoded
            y_train_encoded = np.full(n_kmers, y[i])
            y_train_encoded = label_encoder.transform(y_train_encoded)
            y_train_encoded = to_categorical(y_train_encoded, num_classes=n_classes)

            results = model.evaluate(X_test, y_train_encoded, batch_size=128)
            print("test loss, test acc:", results)
