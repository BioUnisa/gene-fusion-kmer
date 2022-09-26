from encoding import one_hot_encode as ohe
import argparse
import csv
import tensorflow as tf


def read_kmers(input_file):
    # opening the CSV file
    with open(input_file, mode='r') as file:

        csvFile = csv.DictReader(file)
        kmers_list = [] #list
        kmers_encoded = {} #dictionary
        for lines in csvFile:
            X.append(ohe(lines['kmer']))
            y.append(lines['id_gene'])

    return X, y

def training():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='input_file', action='store',
                        default='', help='kmers input file pre_processed')

    args = parser.parse_args()

    read_kmers(args.input_file)

    model = tf.keras.models.load_model('newdat_newmod_jj.h5')
    ##print(model.summary())
    layer = model.get_layer(index=0)

    """
    print(dir(y))
    print(y.filters) # (256)
    print(y.kernel_size) # (20 , 4)
    print(y.activation) # relu
    print(y.input_shape) # (None, 20000, 4, 1)
    print(y.output_shape)
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=787)
    print(X_train.shape, X_train.shape[1:])
    model = tf.keras.models.Sequential()

    model.add(Conv2D(256, kernel_size=(20, 4), activation='relu', input_shape=X_train.shape[:]))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    history = model.fit(X_train, y_train, batch_size=8, epochs=10, validation_data=(X_test, y_test))
    results = model.eveluate(X_test, y_test, batch_size=8)
