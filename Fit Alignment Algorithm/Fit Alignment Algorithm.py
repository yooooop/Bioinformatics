import sys

# Assignment 1
# Yoohyup Lee
# 20934135 
# y342lee

MATCH = 1
MISMATCH = -1
GAP = -1

def score(a, b):
    if a == b:
        return MATCH
    else:
        return MISMATCH

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r") as infile:
        lines = [line.strip() for line in infile if line.strip()]

    seqName2 = lines[2]

    S = lines[1]  
    T = lines[3]

    if seqName2 == ">seq1":
        S = lines[3]
        T = lines[1]

    n = len(S)
    m = len(T)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(0, n):
        dp[i+1][0] = dp[i][0] + GAP  

    for j in range(0, m):
        dp[0][j+1] = 0 

    for i in range(0, n):
        for j in range(0, m):
            dp[i+1][j+1] = max(
                dp[i][j] + score(S[i], T[j]),
                dp[i][j+1] + GAP,
                dp[i+1][j] + GAP
            )

    maxScore = 0
    maxIIndex = n
    maxJIndex = 0

    for j in range(0, m):
        if dp[n][j] > maxScore :
            maxScore = dp[n][j]
            maxJIndex = j

    alignedSeqS = []
    alignedSeqT = []

    while maxIIndex > 0:
        if maxJIndex > 0 and dp[maxIIndex][maxJIndex] == dp[maxIIndex-1][maxJIndex-1] + score(S[maxIIndex-1], T[maxJIndex-1]):
            alignedSeqS.append(S[maxIIndex-1])
            alignedSeqT.append(T[maxJIndex-1])
            maxIIndex -= 1
            maxJIndex -= 1

        elif dp[maxIIndex][maxJIndex] == dp[maxIIndex-1][maxJIndex] + GAP:
            alignedSeqS.append(S[maxIIndex-1])
            alignedSeqT.append('-')
            maxIIndex -= 1

        else:
            alignedSeqS.append('-')
            alignedSeqT.append(T[maxJIndex-1])
            maxJIndex -= 1

    alignedSeqS = ''.join(reversed(alignedSeqS))
    alignedSeqT = ''.join(reversed(alignedSeqT))

    with open(output_file, "w") as outfile:
        outfile.write(f"{maxScore}\n")
        outfile.write(f"{alignedSeqS}\n")
        outfile.write(f"{alignedSeqT}\n")


if __name__ == "__main__":
    main()
