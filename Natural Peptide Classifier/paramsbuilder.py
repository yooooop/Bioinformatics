import math
import random
import itertools
from collections import defaultdict
import tracemalloc
import time

# file to build the parameters necessary for 3-mer and 4-mer

# Yoohyup Lee
# 20934135
# y342lee

K = 4
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
TRAIN_FILE = "uniprot_modified2.fasta"
PARAM_FILE = "params4.txt"

PSEUDOCOUNT = 1  # Laplace smoothing
all_kmers = [''.join(p) for p in itertools.product(ALPHABET, repeat=K)]

def count_kmers(sequences):
    counts = defaultdict(int)
    total = 0

    for seq in sequences:
        for i in range(len(seq) - K + 1):
            kmer = seq[i:i+K]
            counts[kmer] += 1
            total += 1

    return counts, total

# take the sequences in the database and shuffle them to make it random
# and take k-mers from there
def generate_random_sequences(real_sequences):
    random_sequences = []

    for seq in real_sequences:
        shuffled = list(seq)
        random.shuffle(shuffled)
        random_sequences.append("".join(shuffled))

    return random_sequences

def main():
    tracemalloc.start()
    start_time = time.time()

    real_sequences = []
    with open(TRAIN_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                real_sequences.append(line)

    random_sequences = generate_random_sequences(real_sequences)
    real_counts, real_total = count_kmers(real_sequences)
    random_counts, random_total = count_kmers(random_sequences)
    vocab_size = len(all_kmers)

    params = {}

    for kmer in all_kmers:
        real_count = real_counts.get(kmer, 0)
        random_count = random_counts.get(kmer, 0)

        # Laplace smoothing
        p = (real_count + PSEUDOCOUNT) / (real_total + PSEUDOCOUNT * vocab_size)
        q = (random_count + PSEUDOCOUNT) / (random_total + PSEUDOCOUNT * vocab_size)

        params[kmer] = math.log2(p / q)

    with open(PARAM_FILE, "w") as f:
        for kmer, score in params.items():
            f.write(f"{kmer}\t{score}\n")

    end_time = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    building_runtime = end_time - start_time

    print(f"parameter building runtime: {building_runtime:.6f} seconds")
    print(f"parameter building peak memory: {peak_mem / 1024:.2f} KB")


if __name__ == "__main__":
    main()