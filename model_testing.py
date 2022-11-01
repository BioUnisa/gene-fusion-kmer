import argparse
import os
import pickle
import tensorflow as tf

from utils.file import create_conf_path_file, create_test_path_file

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='input_file', action='store',
                        default='kmer-dataset', help='name of input file')
    parser.add_argument('-m', dest='model', action='store',
                        default='./training/gene-fusion-model.h5', help='path of model')

    args = parser.parse_args()

    # Check if input file exists
    conf_path_file = create_conf_path_file(args.input_file)
    test_path_file = create_test_path_file(args.input_file)
    if not os.path.isfile(conf_path_file) or not os.path.isfile(test_path_file):
        print("No such file.")
        print("Please run pre-processing.py first")

    # Load conf object
    conf = {}
    with open(conf_path_file, "rb") as conf_file:
        conf = pickle.load(conf_file)

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