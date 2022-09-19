from Bio import SeqIO
from csv import writer
from functools import partial
from multiprocessing.pool import Pool

import os

K = 15


def build_kmers(sequence, k_size, num_proc):
    kmers = []
    n_kmers = (len(sequence) - k_size + 1)

    # Get start and end offset
    chunk_size = n_kmers // num_proc
    rest_size = n_kmers % num_proc
    rest_added = 0
    offsets = [None] * num_proc
    for i in range(0, num_proc):
        start = i * chunk_size + rest_added
        if rest_size > i:
            offsets[i] = (start, start + chunk_size + 1)
            rest_added += 1
        else:
            offsets[i] = (start, start + chunk_size)

    with Pool(num_proc) as pool:
        for res in pool.map(partial(extract_kmers, sequence, k_size), offsets):
            kmers += res

    return kmers


def extract_kmers(sequence, k_size, offset):
    kmers = []
    start, end = offset

    for i in range(start, end):
        kmer = sequence[i:i + k_size]
        kmers.append(kmer)

    return kmers


def print_gene_kmers(file_path, id_gene, kmers):
    with open(file_path, 'a+') as write_obj:
        csv_writer = writer(write_obj)
        for kmer in kmers:
            csv_writer.writerow([id_gene, kmer])


def generate_kmers_dataset(source_file_path, num_proc, dest_file_path='./training/kmer-dataset.csv'):
    # Create output file
    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
    with open(dest_file_path, 'w') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(['id_gene', 'kmer'])

    # Get kmers for all sequences
    fasta_sequences = SeqIO.parse(open(source_file_path), 'fasta')
    for fasta in fasta_sequences:
        id = fasta.id
        seq = str(fasta.seq)
        kmers = build_kmers(seq, K, num_proc)
        # Print kmers on dataset
        print_gene_kmers(dest_file_path, id, kmers)


if __name__ == '__main__':
    generate_kmers_dataset("./transcripts_genes.fa", 10)
