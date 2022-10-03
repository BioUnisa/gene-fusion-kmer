from Bio import SeqIO
from csv import writer
from functools import partial
from multiprocessing.pool import Pool
from sklearn.model_selection import train_test_split
from typing import Sequence, Tuple

import argparse
import numpy as np
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_sequences_from_fasta(file_path: str) -> (Sequence[str], Sequence[str]):
    _fasta_sequences = SeqIO.parse(open(file_path), 'fasta')
    _id_genes = []
    _sequences = []

    for fasta in _fasta_sequences:
        _id_genes.append(fasta.id)
        _sequences.append(str(fasta.seq))

    return _id_genes, _sequences


def split_sequences_on_processes(_id_genes: Sequence[str], _sequences: Sequence[str],
                                 _num_proc: int) -> Sequence[Sequence[Tuple[str, str]]]:
    # Get size of chunk and rest
    _n_fasta_sequences = len(_sequences)
    _chunk_size = _n_fasta_sequences // _num_proc
    _rest_size = _n_fasta_sequences % _num_proc

    # Initialize arrays to work on
    _split_fasta_sequences = [None] * _num_proc

    rest_added = 0
    for i in range(_num_proc):
        start = i * _chunk_size + rest_added
        if _rest_size > i:
            end = start + _chunk_size + 1
            _split_fasta_sequences[i] = [(_id_genes[i], _sequences[i]) for i in np.arange(start, end)]
            rest_added += 1
        else:
            end = start + _chunk_size
            _split_fasta_sequences[i] = [(_id_genes[i], _sequences[i]) for i in np.arange(start, end)]

    return _split_fasta_sequences


# Create two blank csv file for output
def create_output_files(dest_file_name: str, split: bool) -> (str, str):
    _training_path_file = os.path.join(os.getcwd(), "training", dest_file_name + "-training.csv")
    os.makedirs(os.path.dirname(_training_path_file), exist_ok=True)
    with open(_training_path_file, 'w') as _write_obj:
        _csv_writer = writer(_write_obj)
        _csv_writer.writerow(['id_gene', 'kmer'])

    if split:
        _test_path_file = os.path.join(os.getcwd(), "test", dest_file_name + "-test.csv")
        os.makedirs(os.path.dirname(_test_path_file), exist_ok=True)
        with open(_test_path_file, 'w') as _write_obj:
            _csv_writer = writer(_write_obj)
            _csv_writer.writerow(['id_gene', 'sequence'])
        return _training_path_file, _test_path_file
    else:
        return _training_path_file, ""


def build_kmers(fasta_sequences: Sequence[Tuple[str, str]], k_size: int, file_path: str) -> None:
    for fasta in fasta_sequences:
        id_gene = fasta[0]
        sequence = fasta[1]
        kmers = []
        n_kmers = (len(sequence) - k_size + 1)

        for i in range(n_kmers):
            kmer = sequence[i:i + k_size]
            kmers.append(kmer)

        print_kmers(file_path, id_gene, kmers)


def print_kmers(file_path: str, id_gene: str, kmers: Sequence[int]) -> None:
    with open(file_path, 'a+') as _write_obj:
        _csv_writer = writer(_write_obj)
        for kmer in kmers:
            _csv_writer.writerow([id_gene, kmer])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='input_file', action='store',
                        default='', help='fasta input file')
    parser.add_argument('-k_size', dest='k_size', action='store',
                        type=int, default=15, help='define size of kmer')
    parser.add_argument('-split', dest='split', action='store',
                        type=str2bool, default=True, help='split dataset in training and set')
    parser.add_argument('-t_size', dest='t_size', action='store',
                        type=float, default=0.3, help='define test size')
    parser.add_argument('-num_proc', dest='num_proc', action='store',
                        type=int, default=os.cpu_count(), help='number of processes')
    parser.add_argument('-o', dest='output_file', action='store',
                        default='kmer-dataset', help='name of output file')

    args = parser.parse_args()

    if args.input_file == '':
        print("pre_processing: missing input file")
        print("Try 'pre_processing --help' for more information.")
        exit()

    # Get all id and sequences from input fasta file
    print(f"Start reading {args.input_file} file...")
    id_genes, sequences = read_sequences_from_fasta(args.input_file)
    print(f"Read {len(sequences)} sequence(s)!")

    # Get list of genes
    list_of_genes = np.unique(id_genes)
    print(f"Number of total genes: {len(list_of_genes)}")

    # Split in training and test
    sequences_train = sequences
    id_genes_train = id_genes
    if args.split:
        sequences_train, sequences_test, id_genes_train, id_genes_test = train_test_split(
            sequences, id_genes, test_size=args.t_size, random_state=42)
        print(f"Sequences divided by a coefficient: {args.t_size}")
        print(f"Number of sequences in the training set: {len(sequences_train)}")
        n_test_elements = len(sequences_test)
        print(f"Number of sequences in the test set: {n_test_elements}")
        # Split work on processes
        split_fasta_sequences_train = split_sequences_on_processes(id_genes_train, sequences_train, args.num_proc)
    else:
        # Split work on processes
        split_fasta_sequences_train = split_sequences_on_processes(id_genes, sequences, args.num_proc)

    # Create output files
    training_path_file, test_path_file = create_output_files(args.output_file, args.split)

    # Create kmer training dataset
    with Pool(args.num_proc) as pool:
        pool.map(partial(build_kmers, k_size=args.k_size, file_path=training_path_file), split_fasta_sequences_train)
    print(f"{training_path_file} generated!")

    # Create test dataset
    if args.split:
        with open(test_path_file, 'a+') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerows([[id_genes_test[i], sequences_test[i]] for i in range(n_test_elements)])
        print(f"{test_path_file} generated!")

    # Update counter with number of kmers
    for i in range(len(sequences_train)):
        sequence = sequences_train[i]
        n_kmers = len(sequence) - args.k_size
        for _ in range(n_kmers):
            id_genes_train.append(id_genes_train[i])

    # Save pre-processing information
    import pickle

    config = {
        'k-size': args.k_size,
        'split': args.split
    }
    pickle_path_file = os.path.join(os.getcwd(), "training", args.output_file + ".pickle")
    with open(pickle_path_file, 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
