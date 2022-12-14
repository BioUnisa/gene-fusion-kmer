from csv import writer
from functools import partial
from multiprocessing.pool import Pool

from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Sequence, Tuple, Dict
from utils.file import create_training_path_file, create_test_path_file, create_conf_path_file
from utils.input import str2bool

import argparse
import numpy as np
import os
import pickle


def read_sequences_from_fasta(file_path: str,
                              k_size: int) -> (Sequence[str], Sequence[str], Dict[str, int]):

    _fasta_sequences = SeqIO.parse(open(file_path), 'fasta')
    _id_genes: Sequence[str] = []
    _sequences: Sequence[str] = []
    _map_id_n_kmers: Dict[str, int] = {}

    for fasta in _fasta_sequences:
        _id = fasta.id
        _id_genes.append(_id)
        _sequence = fasta.seq
        _sequences.append(_sequence)
        n_kmers = len(_sequence) - k_size + 1
        if _id not in _map_id_n_kmers:
            _map_id_n_kmers[_id] = n_kmers
        else:
            _map_id_n_kmers[_id] += n_kmers

    return _id_genes, _sequences, _map_id_n_kmers


def split_sequences_on_processes(_id_genes: Sequence[str],
                                 _sequences: Sequence[str],
                                 _num_proc: int) -> Sequence[Sequence[Tuple[str, str]]]:

    # Get size of chunk and rest
    _n_fasta_sequences = len(_sequences)
    _chunk_size = _n_fasta_sequences // _num_proc
    _rest_size = _n_fasta_sequences % _num_proc

    # Initialize arrays to work on
    _split_fasta_sequences: Sequence[str] = [None] * _num_proc

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


def sort_dict(_map_id_n_kmers: dict[str, int]) -> dict[str, int]:
    return {_k: _v for _k, _v in sorted(map_id_n_kmers.items(), key=lambda item: item[1])}


def sample_dataset(_id_genes: Sequence[str],
                   _sequences: Sequence[str],
                   _map_id_n_kmers: dict[str, str],
                   k_size: int,
                   _min: int,
                   _max: int) -> (Sequence[str], Sequence[str], Dict[str, str]):

    n_sequences = len(_sequences)
    new_id_genes: Sequence[str] = []
    new_sequences: Sequence[str] = []
    new_map_id_n_kmers: Dict[str, str] = {_id: 0 for _id in np.unique(id_genes)}

    for i in range(n_sequences):
        _id = _id_genes[i]
        _sequence = _sequences[i]
        if _map_id_n_kmers[_id] >= _min:
            _rest = _max - new_map_id_n_kmers[_id]
            if _rest > 0:
                new_id_genes.append(_id)
                n_kmers = len(_sequence) - k_size + 1
                if n_kmers <= _rest:
                    new_sequences.append(_sequence)
                    new_map_id_n_kmers[_id] += n_kmers
                else:
                    end = k_size + _rest - 1
                    new_sequences.append(_sequence[0:end])
                    new_map_id_n_kmers[_id] += _rest

    return new_id_genes, new_sequences, new_map_id_n_kmers


# Create two blank csv file for output
def create_output_files(file_name: str) -> (str, str):

    _training_path_file = create_training_path_file(file_name)
    os.makedirs(os.path.dirname(_training_path_file), exist_ok=True)
    with open(_training_path_file, 'w') as _write_obj:
        _csv_writer = writer(_write_obj)
        _csv_writer.writerow(['id_gene', 'kmer'])

    _test_path_file = create_test_path_file(file_name)
    os.makedirs(os.path.dirname(_test_path_file), exist_ok=True)
    with open(_test_path_file, 'w') as _write_obj:
        _csv_writer = writer(_write_obj)
        _csv_writer.writerow(['id_gene', 'sequence'])
    return _training_path_file, _test_path_file


def build_kmers(fasta_sequences: Sequence[Tuple[str, str]],
                k_size: int,
                file_path: str) -> None:

    for fasta in fasta_sequences:
        id_gene = fasta[0]
        sequence = fasta[1]
        kmers = []
        n_kmers = len(sequence) - k_size + 1

        for i in range(n_kmers):
            kmer = sequence[i:i + k_size]
            kmers.append(kmer)

        print_kmers(file_path, id_gene, kmers)


def print_kmers(file_path: str,
                id_gene: str,
                kmers: Sequence[int]) -> None:

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
    parser.add_argument('-min', dest='min', action='store',
                        type=int, default=0, help='min number of kmers that genes need to have')
    parser.add_argument('-max', dest='max', action='store',
                        type=int, default=np.inf, help='max number of kmers that genes need to have')
    parser.add_argument('-t_size', dest='t_size', action='store',
                        type=float, default=0.2, help='define test size')
    parser.add_argument('-num_proc', dest='num_proc', action='store',
                        type=int, default=os.cpu_count(), help='number of processes')
    parser.add_argument('-o', dest='output_file', action='store',
                        default='kmer-dataset', help='name of output file')

    args = parser.parse_args()

    if args.input_file == '':
        print('pre_processing: missing input file')
        print('Try "pre_processing.py" --help for more information.')
        exit()

    # Get all id and sequences from input fasta file
    print(f'Start reading {args.input_file} file...')
    id_genes, sequences, map_id_n_kmers = read_sequences_from_fasta(args.input_file, args.k_size)
    print(f'Read {len(sequences)} sequence(s)!')
    # Log number of kmers for each gene
    map_id_n_kmers = sort_dict(map_id_n_kmers)
    with open('pre_processing.log', 'w') as logger:
        logger.write("Before performing sampling operations\n")
        for k, v in map_id_n_kmers.items():
            logger.write(f'{k} : {v}\n')

    # Sample dataset with min and max
    id_genes, sequences, map_id_n_kmers = sample_dataset(
        id_genes, sequences, map_id_n_kmers, args.k_size, args.min, args.max)
    # Log number of kmers for each gene
    map_id_n_kmers = sort_dict(map_id_n_kmers)
    with open('pre_processing.log', 'a') as logger:
        logger.write("\nAfter performing sampling operations\n")
        for k, v in map_id_n_kmers.items():
            if v > 0:
                logger.write(f'{k} : {v}\n')
    print("Sampling phase terminated")
    print("pre_processing.log generated!")

    # Get label and encoded label
    classes = np.unique(id_genes)
    n_classes = len(classes)
    print(f'Number of total genes: {n_classes}')
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(id_genes)

    # Split in training and test
    sequences_train = sequences
    id_genes_train = id_genes
    sequences_train, sequences_test, id_genes_train, id_genes_test = train_test_split(
        sequences, id_genes, test_size=args.t_size, random_state=42)
    print(f'Sequences divided by a coefficient: {args.t_size}')
    print(f'Number of sequences in the training set: {len(sequences_train)}')
    n_test_elements = len(sequences_test)
    print(f'Number of sequences in the test set: {n_test_elements}')
    # Split work on processes
    split_fasta_sequences_train = split_sequences_on_processes(
        id_genes_train, sequences_train, args.num_proc)

    # Create output files
    training_path_file, test_path_file = create_output_files(args.output_file)

    # Create kmer training dataset
    with Pool(args.num_proc) as pool:
        pool.map(partial(build_kmers, k_size=args.k_size, file_path=training_path_file),
                 split_fasta_sequences_train)
    print(f'{training_path_file} generated!')

    # Create test dataset
    with open(test_path_file, 'a+') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerows([[id_genes_test[i], sequences_test[i]] for i in range(n_test_elements)])
    print(f'{test_path_file} generated!')

    # Save pre-processing information
    config = {
        'k-size': args.k_size,
        'classes': classes,
        'n_classes': n_classes,
        'label-encoder': label_encoder
    }
    conf_path_file = create_conf_path_file(args.output_file)
    with open(conf_path_file, 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
