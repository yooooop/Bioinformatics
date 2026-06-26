import argparse
import torch
import torch.nn as nn
import numpy as np
import re
import logging
import random

from matchms.importing.load_from_msp import parse_msp_file, parse_spectrum_dict
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logging.getLogger("matchms").setLevel(logging.ERROR)

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i + 1 for i, a in enumerate(AA_LIST)}


# this is the same residual block as the prediction model
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

# this is also the same spectrum model as the prediction model
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# build intensities matrixes to train on of b y ions
def extract_target(spec, seq):
    
    L = len(seq)
    target = np.zeros((L - 1, 2), dtype=np.float32)

    intensities = spec.peaks.intensities
    mz_values = spec.peaks.mz
    peak_comments = spec.metadata.get("peak_comments")

    if peak_comments is None:
        return None

    mz_keys = list(peak_comments.keys())

    for i, intensity in enumerate(intensities):
        mz = mz_values[i]

        # match peak to closest annotated m/z
        closest_mz = min(mz_keys, key=lambda x: abs(x - mz))
        ann = peak_comments[closest_mz] if abs(closest_mz - mz) < 1e-4 else None

        if ann is None or not isinstance(ann, str):
            continue

        ann = ann.strip('"')

        for entry in ann.split(","):
            entry = entry.split("/")[0]

            # parse ion format as not all of them just annotated simply like b3 y5
            match = re.match(r'([by])(\d+)(?:\^(\d+))?', entry)
            if not match:
                continue

            ion_type = match.group(1)
            index = int(match.group(2))
            charge = int(match.group(3)) if match.group(3) else 1

            # only keep singly charged ions
            if charge != 1:
                continue

            if ion_type == "b" and 1 <= index <= L - 1:
                target[index - 1, 0] = max(target[index - 1, 0], intensity)
            elif ion_type == "y" and 1 <= index <= L - 1:
                site = L - index - 1
                target[site, 1] = max(target[site, 1], intensity)

    # normalize intensities using the highest intensity
    M = target.max()
    if M > 0:
        target /= M

    return target


def load_dataset(msp_file):
    dataset = []
    spectra = parse_msp_file(msp_file)

    while True:
        # only really here with this weird loop because stuff seemed to break without it
        # my guess is that some sequences in the library may be slightly broken idk
        try:
            spectrum_dict = next(spectra)
        except StopIteration:
            break
        except Exception:
            continue

        try:
            spec = parse_spectrum_dict(
                spectrum=spectrum_dict,
                metadata_harmonization=True,
                spectrum_type="own"
            )
        except Exception:
            continue

        name = spec.metadata.get("compound_name")
        if name is None:
            continue

        # extract sequence and charge from name
        seq = name.split("/")[0]
        match = re.search(r"/(\d+)", name)
        charge = int(match.group(1)) if match else 1

        # filter by ones with the 20 amino acids and lengths from 5 to 30 only
        if len(seq) < 5 or len(seq) > 30:
            continue
        if any(a not in AA_TO_IDX for a in seq):
            continue

        # only use ones with C modifications
        if seq.count("C") != name.count("CAM"):
            continue

        target = extract_target(spec, seq)
        if target is None:
            continue

        # convert to tensors
        x = torch.tensor([AA_TO_IDX[a] for a in seq], dtype=torch.long)
        y = torch.tensor(target, dtype=torch.float32)

        dataset.append((x, y))

    print(f"Loaded {len(dataset)} spectra")

    return dataset

# for padding variable-length sequences
def collate_batch(batch):
    xs, ys = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0)

    max_len = max(y.shape[0] for y in ys)
    y_pad, mask = [], []

    for y in ys:
        pad_len = max_len - y.shape[0]

        if pad_len > 0:
            mask.append(torch.cat([torch.ones(y.shape[0]), torch.zeros(pad_len)]))
            y = torch.cat([y, torch.zeros(pad_len, 2)], dim=0)
        else:
            mask.append(torch.ones(y.shape[0]))

        y_pad.append(y)

    return x_pad, torch.stack(y_pad), torch.stack(mask)

# training function
def train(args):
    dataset = load_dataset(args.msp_file)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_batch
    )

    # used for 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrumModel().to(device)

    print("Parameters:", count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.MSELoss(reduction='none')

    for epoch in range(args.epoch):
        total_loss = total_mse = total_acc = n = 0

        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            # Forward pass + sigmoid for normalized outputs
            pred = torch.sigmoid(model(x))

            # Align sequence lengths
            min_len = min(pred.shape[1], y.shape[1], mask.shape[1])
            pred, y = pred[:, :min_len], y[:, :min_len]
            mask = mask[:, :min_len].unsqueeze(-1)

            denom = mask.sum()
            if denom == 0:
                continue

            # masked loss while ignoring padding
            loss = loss_fn(pred, y)
            loss = (loss * mask).sum() / denom

            # compute metrics on valid positions only
            valid_idx = mask.squeeze(-1).bool()
            pred_flat, y_flat = pred[valid_idx], y[valid_idx]

            mse = F.mse_loss(pred_flat, y_flat)

            # binary accuracy using peaks
            pred_binary = (pred_flat > 1e-2).float()
            y_binary = (y_flat > 1e-2).float()
            acc = (pred_binary == y_binary).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse.item()
            total_acc += acc.item()
            n += 1

        print(f"Epoch {epoch+1}: "f"loss={total_loss/n:.6f}, "f"MSE={total_mse/n:.6f}, "f"binary_acc={total_acc/n:.4f}")

    torch.save(model.state_dict(), args.param_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("msp_file")
    parser.add_argument("param_file")
    parser.add_argument("--epoch", type=int, default=20)

    args = parser.parse_args()

    train(args)