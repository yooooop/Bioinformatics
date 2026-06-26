import sys
import random

# file to sample natural and random/shuffled substrings from an actual database

# Yoohyup Lee
# 20934135
# y342lee

# length/max length to generate
SUBSTRING_LENGTH = 40
# N value
NUM_SUBSTRINGS = 500


def main():
    if len(sys.argv) != 2:
        print("Usage: python sample_substrings.py input.txt")
        sys.exit(1)

    input_file = sys.argv[1]

    # Take in all long enough sequences
    sequences = []
    with open(input_file, "r") as f:
        for line in f:
            seq = line.strip()
            if len(seq) >= SUBSTRING_LENGTH:
                sequences.append(seq)

    sampledNatural = []
    sampledRandom = []

    # take a random sequence from the file, and take a 20-40 length substring from a random position
    while len(sampledNatural) < NUM_SUBSTRINGS:
        seq = random.choice(sequences)
        randLength = random.randint(20, SUBSTRING_LENGTH)
        start = random.randint(0, len(seq) - SUBSTRING_LENGTH)
        substring = seq[start:start + randLength]
        sampledNatural.append(substring)

    with open("naturalpeptides.txt", "w") as out:
        for s in sampledNatural:
            out.write(s + "\n")

    # take a random sequence from the file and take a 20-40 length substring from a random position and shuffle that substring
    while len(sampledRandom) < NUM_SUBSTRINGS:
        seq = random.choice(sequences)
        randLength = random.randint(20, SUBSTRING_LENGTH)
        start = random.randint(0, len(seq) - SUBSTRING_LENGTH)
        substring = seq[start:start + randLength]
        shuffled = "".join(random.sample(substring, len(substring)))
        sampledRandom.append(shuffled)

    with open("randompeptides.txt", "w") as out:
        for s in sampledRandom:
            out.write(s + "\n")


if __name__ == "__main__":
    main()