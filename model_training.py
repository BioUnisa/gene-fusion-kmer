import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.backend import image_data_format
from keras.layers import Embedding, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from training.encoding import one_hot_encode as ohe


def encode2train(input_file):
    df = pd.read_csv(input_file, header=0, engine="python")
    df_seq = df.iloc[:, 1]

    X = np.array([ohe(x) for x in df_seq])
    y = np.array(df.iloc[:, 0])

    print(X.shape, y.shape)

    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='input_file', action='store',
                        default='./training/kmer-dataset.csv', help='kmers input file pre_processed')

    args = parser.parse_args()

    X, y = encode2train(args.input_file)

    print(y)
    pippo = LabelEncoder()
    vec = pippo.fit_transform(y)
    vec = to_categorical(vec)

    Nsamp, LEN_SEQ, WID_SEQ = X.shape
    if image_data_format() == "channels_first":
        X = X.reshape(Nsamp, 1, LEN_SEQ, WID_SEQ)
    else:
        X = X.reshape(Nsamp, LEN_SEQ, WID_SEQ, 1)

    print("Dimension of data:" + str(X.shape))

    # model = tf.keras.models.load_model('newdat_newmod_jj.h5')
    # print(model.summary())
    # layer = model.get_layer(index=0)

    X_train, X_test, y_train, y_test = train_test_split(X, vec, test_size=0.3, random_state=42)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = tf.keras.models.Sequential()

    #model.add(Embedding(4, output_dim=8, input_length=15))
    model.add(Conv2D(128, (3, 2), activation='relu', input_shape=X.shape[1:]))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 2), activation='relu', input_shape=X.shape[1:]))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (2, 1), activation='relu', input_shape=X.shape[1:]))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(106, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()

    history = model.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))
    results = model.evaluate(X_test, y_test, batch_size=128)