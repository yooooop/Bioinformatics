# Bioinformatics Projects

This repository contains several independent mini-projects completed for bioinformatics coursework and research.

---

# 1. Fit Alignment Algorithm

## Running the Program

To run this code, the first argument should be the file’s name, **assn1-part1.py** and then the fasta file that you want the sequence from and then the out text file that the answer will be written to.

Example:

```bash
python assn1-part1.py in.fasta out.txt
```

The rest follow all rules from the assignment descriptions.

## Implementation

### Line 14-30

The code first starts by getting all necessary variables needed to calculate the fit alignment score and its sequence.

For this, we need to first read the file from the command line. Then, we check if the second sequence given in the file is titled **seq1**. If it is, then we set that as **S** and its length **n** with the other sequence automatically set as **T** and **m** respectively. Otherwise, the first sequence will always be considered **S** and its length as **n** (and the second sequence as **T** and **m** respectively).

### Line 32-46

This is where we set the DP table and fill in the values.

This mixes local alignment and global alignment. For aligning **S**, we need to globally fit the entire sequence in and so the first column is filled with an increasing gap penalty (-1, -2, -3, …). However, as deleting from **T** doesn’t give you a penalty (like local alignment), the first row is filled with 0s.

Then, we fill the rest of the table by using the scoring scheme given by the assignment just like how other alignment DP tables are filled in. Compare the value from the left + gap penalty, up + gap penalty, and left-up + the score from mismatch or match and take whatever is the maximum value.

### Line 48-83

After filling the table, we can reconstruct the sequences by backtracking.

Since we have to use all of **S**, we start at the last column of the table but since we don’t have to use all of **T**, we start at the maximum value at the last column of the table.

We move up one by one, adding a letter or a gap to each **S** and **T** each time we move. The direction we move is determined by checking if the score to the left + gap, up + gap, or left-up + score is equal to the score of the current cell prioritizing moving left-up, left, up in that order (tiebreaking).

After appending the letters to the sequence, since we went backwards, reverse the string and then output the maxScore (the max value in the last column) and the two strings calculated.

## Complexity

**Time Complexity**

O(nm), where **n** and **m** are the lengths of sequences **S** and **T** respectively.

This is because we calculate the table that is (n+1) × (m+1) once by doing O(1) operations. Backtracking is linear time as we will only ever move up n+m cells computing the final sequences.

(n+m < nm)

**Space Complexity**

O(nm), since we need a DP table of size (n+1) × (m+1). Rebuilding the string only requires O(n+m) space.

---

# 2. Natural Peptide Classifier

## Running the Program

To run:

```bash
python assn2-part1.py <input_file> <output_file>
```

with **params3.txt** and **params4.txt** in the same folder.

## Overview

The first scoring function uses 3-mer sequencing.

The second scoring function uses 4-mer sequencing.

There was an attempt to make it ML but actually had lower accuracy than simply using a 4-mer scoring function.

To build the parameters for the 3-mers and 4-mers, **paramsbuilder.py** was used to first collect all the sequences in the Uniprot Reviewed database and calculate the frequencies of each k-mer (this is where p comes from).

For the q value, we take all sequences from the database and shuffle them. Then, we also take the frequencies of each k-mer in the random sequences as well.

Laplace smoothing is also added to account for k-mers that may appear 0 times.

Since these are precomputed before the main code runs, the database used is not included in the submission (it isn’t necessary to run the program but the frequencies were sampled from the entire Uniprot reviewed database but the parameter building runtime and space usage is put in **testing_result.txt**).

Please check out **testing_result.txt** for runtime and memory usage for each of the scoring functions.

## Dataset

500 natural peptides used to test were sampled from the Uniprot Reviewed (SwissProt) database.

Note that the k-mer frequencies calculated in **params3.txt** and **params4.txt** were also obtained from the same database (the "p" variable from the log2(p/q) formula).

The sequences were obtained using **databasemodifier.py** which takes in a fasta file and gets rid of lines starting with >.

Then, it goes through **naturalpeptidessampler.py** to generate N amount of natural and random peptides (random peptides were obtained by shuffling each natural peptide).

For the parameters, they are calculated using **paramsbuilder.py**.

All the files mentioned are attached in the submission except for the actual database which is available as Reviewed (Swiss-Prot) in uniprot.org/downloads.

## Results

The first scoring function using a 3-mer had the accuracy of **300/500** where N = 500 (i.e. 261 peptides were in the top 500 in terms of score) in my testing file.

The accuracy of the second scoring function with 4-mers had the accuracy of **60–70%** and an accuracy of **329/500** on my testing file.

The first 500 sequences in the test file are natural and the other 500 are random (so that it was easy to rank).

---

# 3. Peptide MS/MS Spectrum Prediction

The model consists of a deep convolutional neural network trained on peptide data derived from the NIST peptide spectral library using approximately **1,721,730 parameters** from the human dataset.

The model’s goal is to predict fragment ion intensities for sequences of **5 to 30 amino acids** in length with CAM/Cysteine modifications or no modifications only.

Each peptide sequence is first converted into a **128-dimensional learned embedding representation** (using 20 standard amino acids + 1 padding token).

The embedded sequence is then processed through convolutional layers with different kernel sizes (2, 3, 5, 7).

The outputs from these layers are then concatenated and go through a stack of 4 residual blocks.

The model incorporates a dropout rate of **0.2** after the residual layers which reduces overfitting.

Finally, the network produces predictions for each b/y ion using a linear output layer, with a sigmoid activation function applied to it.

## Data Processing

The data is parsed/extracted with the **matchms** library in Python.

The sequences are filtered according to the rules stated by the assignment:

* Length of 5 to 30
* CAM modifications with C only
* Other amino acids unmodified

For each intensity and b/y ion, only ions with charge **+1** are used.

Multiple annotations per peak were handled by matching the nearest m/z within the tolerance of **1e-4**.

The intensities are then normalized with their maximum intensity so that all intensities are between 0 and 1.

Also, any value below the threshold of **1e-2** is discarded/set to 0.

## Additional Information

Please look at the report in the **Peptide MS/MS Spectrum Prediction** folder.
