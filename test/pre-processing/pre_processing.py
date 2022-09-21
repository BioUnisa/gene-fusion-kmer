from Bio import SeqIO
from csv import writer

import argparse
import os


def generate_kmers_dataset(source_file_path, k_size, dest_file_path):
    # Create output file
    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
    with open(dest_file_path, 'w') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(['id_gene', 'kmer'])

    # Get kmers for all sequences
    fasta_sequences = SeqIO.parse(open(source_file_path), 'fasta')
    for fasta in fasta_sequences:
        id = fasta.id
        sequence = str(fasta.seq)
        kmers = []
        n_kmers = (len(sequence) - k_size + 1)

        for i in range(n_kmers):
            kmer = sequence[i:i + k_size]
            kmers.append(kmer)

        print_gene_kmers(dest_file_path, id, kmers)


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
    parser.add_argument('-o', dest='output_file', action='store',
                        default='./training/kmer-dataset.csv', help='output dataset file')

    args = parser.parse_args()

    if args.input_file == '':
        print("pre_processing: missing input file")
        print("Try 'pre_processing --help' for more information.")
        exit()

    generate_kmers_dataset(args.input_file, args.k_size, args.output_file)
