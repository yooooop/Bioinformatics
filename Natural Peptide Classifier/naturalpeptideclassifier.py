import sys
import time
import tracemalloc

# the actual file with the model and the 3-mer scoring system

# Yoohyup Lee
# 20934135
# y342lee

K3 = 3
PARAM_FILE3 = "params3.txt"
K4 = 4
PARAM_FILE4 = "params4.txt"

MAX_LEN = 40

###################################################################
# Functions for 3-mer scoring

def load_params(filename):
    params = {}
    with open(filename) as f:
        for line in f:
            kmer, score = line.strip().split()
            params[kmer] = float(score)
    return params


# only thing I need to do is compare the frequencies from the 
# parameters and add to sum
def score_peptide(peptide, params, K):
    score = 0.0
    count = 0

    peptide = peptide.upper()

    for i in range(len(peptide) - K + 1):
        kmer = peptide[i:i+K]
        score += params.get(kmer, 0.0)
        count += 1

    if count > 0:
        score /= count   # normalize by number of kmers

    return score

###################################################################

def count_ids_below_500(scoreresults, top_n=500): 
    count = 0 
    i = 0
    for line in scoreresults: 
        seq_id = int(line[0])  
        if i >= top_n: 
            break
        if seq_id < 500: 
            count += 1 
        i += 1
    return count

def main():
    # USED in.fasta to test, and written to out.txt
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    params = load_params(PARAM_FILE3)

    peptides = []
    with open(input_file) as f:
        for line in f:
            peptide = line.strip()
            if peptide:
                peptides.append(peptide)

    # scoring function 1, 3-mer
    tracemalloc.start()
    start_time = time.time()

    score1_results = []
    score1compare = []
    id = 1
    for peptide in peptides:
        s1 = score_peptide(peptide, params, K3)
        score1_results.append(s1)
        score1compare.append((id, s1))
        id += 1

    score1compare.sort(key=lambda x: x[1], reverse=True)

    score1 = count_ids_below_500(score1compare)
    print(score1)

    end_time = time.time()
    _, peak_mem1 = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    score1_runtime = end_time - start_time

    print(f"3-mer runtime: {score1_runtime:.6f} seconds")
    print(f"3-mer peak memory: {peak_mem1 / 1024:.2f} KB")

    # scoring function 2, 4-mer
    tracemalloc.start()
    start_time = time.time()

    score2_results = []
    score2compare = []
    id = 1
    params = load_params(PARAM_FILE4)

    for peptide in peptides:
        s2 = score_peptide(peptide, params, K4)
        score2_results.append(s2)
        score2compare.append((id, s2))
        id += 1

    score2compare.sort(key=lambda x: x[1], reverse=True)

    score2 = count_ids_below_500(score2compare)
    print(score2)

    end_time = time.time()
    _, peak_mem2 = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"4-mer runtime: {score1_runtime:.6f} seconds")
    print(f"4-mer peak memory: {peak_mem1 / 1024:.2f} KB")

    score2_runtime = end_time - start_time

    print(f"Score2 runtime: {score2_runtime:.6f} seconds")
    print(f"Score2 peak memory: {peak_mem2 / 1024:.2f} KB")

    with open(output_file, "w") as f:
        for peptide, s1, s2 in zip(peptides, score1_results, score2_results):
            f.write(f"{s1:.6f} {s2:.6f} {peptide}\n")
        

    # parameter building time added after the code was run    
    with open("testing_results.txt", "w") as f:
        f.write("=== Score1 (3-mer model) ===\n")
        f.write(f"Runtime: {score1_runtime:.6f} seconds\n")
        f.write(f"Peak memory: {peak_mem1 / 1024:.2f} KB\n")
        f.write(f"accuracy: {score1}/500\n\n")

        f.write("=== Score2 (4-mer model) ===\n")
        f.write(f"Runtime: {score2_runtime:.6f} seconds\n")
        f.write(f"Peak memory: {peak_mem2 / 1024:.2f} KB\n")
        f.write(f"accuracy: {score2}/500\n")

if __name__ == "__main__":
    main()
