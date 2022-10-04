from Bio import SeqIO
from typing import Sequence

import os


def read_sequences_from_fasta(file_path: str) -> (Sequence[str], Sequence[str]):
    _fasta_sequences = SeqIO.parse(open(file_path), 'fasta')
    _id_genes = []
    _sequences = []

    for fasta in _fasta_sequences:
        _id_genes.append(fasta.id)
        _sequences.append(str(fasta.seq))

    return _id_genes, _sequences


def create_training_path_file(file_name: str) -> str:
    _training_path_file = os.path.join(os.getcwd(), "training", file_name + "-training.csv")
    return _training_path_file


def create_test_path_file(file_name: str) -> str:
    _test_path_file = os.path.join(os.getcwd(), "test", file_name + "-test.csv")
    return _test_path_file


def create_conf_path_file(file_name: str) -> str:
    _conf_path_file = os.path.join(os.getcwd(), "training", file_name + ".pickle")
    return _conf_path_file
