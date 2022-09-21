#!/bin/bash

pre_processing="./test/pre-processing/pre_processing.py"
pre_processing_parallel="./pre_processing.py"
fasta_file="./test/pre-processing/transcripts_genes.fa"

oracle="./test/pre-processing/oracle.csv"
oracle_sorted="./test/pre-processing/oracle-sorted.csv"
output="./test/pre-processing/output.csv"
output_sorted="./test/pre-processing/output_sorted.csv"

NP=32

rm $oracle 2>/dev/null
rm $oracle_sorted 2>/dev/null
rm $output 2>/dev/null
rm $output_sorted 2>/dev/null

python3 $pre_processing -f $fasta_file -o $oracle
sort $oracle >$oracle_sorted
rm $oracle

for i in $(seq 1 $NP); do

  python3 $pre_processing_parallel -f $fasta_file -o $output -num_proc "$i" 1>/dev/null
  sort $output >$output_sorted
  rm $output

  if cmp -s "$output_sorted" "$oracle_sorted"; then
    echo -ne "${i}/${NP} - Test passed!\r"
  else
    echo "${i}/${NP} - Test not passed!"
  fi

  rm $output_sorted

done

rm $oracle_sorted

echo "All tests have been completed!"
