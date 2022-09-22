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
            #print(lines['id_gene'])
            kmers_list.append((lines['id_gene'],lines['kmer']))

    for elem in kmers_list:
        kmers_encoded[elem[0]] = ohe(elem[1])
        #print(kmers_encoded)
    return kmers_encoded


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='input_file', action='store',
                        default='', help='kmers input file pre_processed')

    args = parser.parse_args()

    read_kmers(args.input_file)

    #model = tf.keras.models.load_model('newdat_newmod_jj.h5')
    #print(model.summary())

