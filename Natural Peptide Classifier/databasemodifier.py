import sys

# file to modify a fasta file/database so that the database doesn't contain any 
# "unnatural" peptides (ones that aren't the 20 regular ones)
# and also gets rid of any lines that isn't a sequence (starts with >)

# Yoohyup Lee
# 20934135
# y342lee

def main():
    if len(sys.argv) != 3:
        print("Usage: python flatten_fasta.py input.fasta output.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        sequence_parts = []

        for line in fin:
            line = line.strip()

            if not line:
                continue
            
            # if the line is a header, get rid of it and append the previous sequence into the database
            if line.startswith(">"):
                if sequence_parts:
                    # take all the lines collected and put them as one line into the modified database
                    fout.write("".join(sequence_parts) + "\n")
                    sequence_parts = []
            # ignore any sequence with amino acids that aren't the standard 20
            elif line.__contains__("B"):
                continue
            elif line.__contains__("J"):
                continue
            elif line.__contains__("O"):
                continue
            elif line.__contains__("U"):
                continue
            elif line.__contains__("X"):
                continue
            elif line.__contains__("Z"):
                continue
            else:
                sequence_parts.append(line)

        # write the last line in
        if sequence_parts:
            fout.write("".join(sequence_parts) + "\n")


if __name__ == "__main__":
    main()
