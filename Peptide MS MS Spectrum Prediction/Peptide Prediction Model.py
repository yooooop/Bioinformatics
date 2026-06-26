import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i + 1 for i, a in enumerate(AA_LIST)}

def encode_sequence(seq):
    return torch.tensor([AA_TO_IDX[a] for a in seq], dtype=torch.long)

# this is the same residual block as the training model
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # first convolution layer
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        
        # second convolution layer
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        # apply two convolution layers
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        # add original input back (skip connection)
        x = x + identity
        return self.relu(x)

# this is also the same spectrum model as the training model
class SpectrumModel(nn.Module):

    def __init__(self):
        super().__init__()

        embed_dim = 128
        self.embedding = nn.Embedding(len(AA_LIST) + 1, embed_dim, padding_idx=0)
        # varying kernel sizes
        self.conv2 = nn.Conv1d(embed_dim, 64, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3)
        self.relu = nn.ReLU()

        # after concatentation
        merged = 64 * 4
        # stack layers
        self.res1 = ResidualBlock(merged)
        self.res2 = ResidualBlock(merged)
        self.res3 = ResidualBlock(merged)
        self.res4 = ResidualBlock(merged)
        # overfitting prevention       
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(merged, 2)

    def forward(self, x):

        # using different kernel sizes of 2,3,5,7
        x = self.embedding(x)
        x = x.transpose(1, 2)
        c2 = self.relu(self.conv2(x))
        c3 = self.relu(self.conv3(x))
        c5 = self.relu(self.conv5(x))
        c7 = self.relu(self.conv7(x))
        # crop before concatenating
        min_len = min(c2.shape[2], c3.shape[2], c5.shape[2], c7.shape[2])

        c2 = c2[:, :, :min_len]
        c3 = c3[:, :, :min_len]
        c5 = c5[:, :, :min_len]
        c7 = c7[:, :, :min_len]

        x = torch.cat([c2, c3, c5, c7], dim=1)
        # residual blocks used to learn better
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        # applying dropout
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.output(x)

        return x

# finds the sequence and charge per line and returns it
def parse_line(line):
    parts = line.strip().split(",")
    seq = parts[0].split("=")[1]
    charge = int(parts[1].split("=")[1])
    return seq, charge

# def load_actual(jsonl_file):

#     actual_dict = {}

#     with open(jsonl_file) as f:
#         for line in f:
#             data = json.loads(line)

#             key = (data["sequence"], data["precursor_charge"])
#             actual_dict[key] = data

#     return actual_dict

# def cosine_similarity(a, b):
#     a = a.flatten()
#     b = b.flatten()

#     if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
#         return 0.0

#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# def plot_spectrum(pred, true, seq, charge, name):

#     pred = np.array(pred)
#     true = np.array(true)

#     min_len = min(len(pred), len(true))
#     pred = pred[:min_len]
#     true = true[:min_len]

#     x = np.arange(min_len)

#     sim = cosine_similarity(pred, true)

#     plt.figure(figsize=(14, 5))

#     plt.vlines(x, 0, true[:,0], linewidth=2, label="Exp b-ions")
#     plt.vlines(x, 0, true[:,1], linewidth=2, colors='orange', label="Exp y-ions")

#    plt.vlines(x, 0, -pred[:,0], linestyles='dashed', label="Pred b-ions")
#    plt.vlines(x, 0, -pred[:,1], linestyles='dashed', colors='green', label="Pred y-ions")

#    plt.axhline(0, color='black', linewidth=1)

#    plt.title(f"{seq}, charge {charge}, similarity {sim:.4f}\n{name}")

#    plt.xlabel("Cleavage Site")
#    plt.ylabel("Intensity")

#    plt.legend()
#    plt.tight_layout()
#    plt.show()


############################################################
# Prediction + Plotting
############################################################

def predict(seq_file, out_file, param_file):

    # use gpu when available, else cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = SpectrumModel()
    model.load_state_dict(torch.load(param_file, map_location=device))
    model.to(device)
    model.eval()

    # all lines used to plot the spectrum to compare tot actual results are commented out (including functions used)
    # to use it to plot, you can do it by setting the actual_file variable to a jsonl file with actual real data in it
    # actual_data = load_actual(actual_file)

    # for each sequence in the test_input file
    with open(seq_file) as fin, open(out_file, "w") as fout:
        for line in fin:
            if line.strip() == "":
                continue

            seq, charge = parse_line(line)

            # key = (seq, charge)

            # if key not in actual_data:
            #    print(f"Skipping {seq}, no ground truth found")
            #    continue

            # true = actual_data[key]["sites"]
            # name = f"{seq}/{charge}"

            # encode sequence before it's available for a prediction
            x = encode_sequence(seq).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = torch.sigmoid(model(x))[0]

            pred = pred[:len(seq)-1]
            pred = pred.cpu().numpy()
            # if the value predicted is less than the threshold, set to 0
            pred[pred < 1e-2] = 0.0
            
            # plot_spectrum(pred, true, seq, charge, name)

            result = {
                "sequence": seq,
                "precursor_charge": charge,
                "length": len(seq),
                "sites": pred.tolist()
            }

            fout.write(json.dumps(result) + "\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("seq_file")
    parser.add_argument("out_file")
    parser.add_argument("--param_file", default="trained_model.pth")

    args = parser.parse_args()

    predict(args.seq_file, args.out_file, args.param_file)
