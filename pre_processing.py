from Bio import SeqIO
from csv import writer
from functools import partial
from multiprocessing.pool import Pool

import argparse
import os


def generate_kmers_dataset(source_file_path, k_size, dest_file_path, num_proc):
    # Read all fasta
    _fasta_sequences = SeqIO.parse(open(source_file_path), 'fasta')
    print(f"Start reading {source_file_path} file...")
    fasta_sequences = []
    for fasta in _fasta_sequences:
        id_gene = fasta.id
        sequence = str(fasta.seq)
        fasta_sequences.append((id_gene, sequence))

    # Create output file
    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
    with open(dest_file_path, 'w') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(['id_gene', 'kmer'])

    # Split fasta sequences
    n_sequences = len(fasta_sequences)
    chunk_size = n_sequences // num_proc
    rest_size = n_sequences % num_proc
    splitted_fasta_sequences = [None] * num_proc
    rest_added = 0
    for i in range(num_proc):
        start = i * chunk_size + rest_added
        if rest_size > i:
            splitted_fasta_sequences[i] = fasta_sequences[start:start + chunk_size + 1]
            rest_added += 1
        else:
            splitted_fasta_sequences[i] = fasta_sequences[start:start + chunk_size]

    with Pool(num_proc) as pool:
        pool.map(partial(build_kmers, k_size=k_size, file_path=dest_file_path), splitted_fasta_sequences)


def build_kmers(fasta_sequences, k_size, file_path):
    for fasta in fasta_sequences:
        id_gene = fasta[0]
        sequence = fasta[1]
        kmers = []
        n_kmers = (len(sequence) - k_size + 1)

        for i in range(n_kmers):
            kmer = sequence[i:i + k_size]
            kmers.append(kmer)

        print_gene_kmers(file_path, id_gene, kmers)


def print_gene_kmers(file_path, id_gene, kmers):
    with open(file_path, 'a+') as write_obj:
        csv_writer = writer(write_obj)
        for kmer in kmers:
            csv_writer.writerow([id_gene, kmer])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='input_file', action='store',
                        default='', help='fasta input file')
    parser.add_argument('-k_size', dest='k_size', action='store',
                        type=int, default=15, help='define size of kmer')
    parser.add_argument('-num_proc', dest='num_proc', action='store',
                        type=int, default=os.cpu_count(), help='number of processes')
    parser.add_argument('-o', dest='output_file', action='store',
                        default='./training/kmer-dataset.csv', help='output dataset file')

    args = parser.parse_args()

    if args.input_file == '':
        print("pre_processing: missing input file")
        print("Try 'pre_processing --help' for more information.")
        exit()

    generate_kmers_dataset(args.input_file, args.k_size, args.output_file, args.num_proc)
    print(f"{args.output_file} generated!")
