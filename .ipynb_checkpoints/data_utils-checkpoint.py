"""
Shared utilities for CriDe training scripts (BiLSTM + Transformer).
Put this file in the same folder as your training notebooks/scripts so
`from data_utils import ...` works.

Needs: torch, numpy, scikit-learn
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class UCFCrimeDataset(Dataset):
    """
    Expects base_dir/<class_name>/*.npy, each holding a
    [num_segments, feature_dim] array for one video.
    num_segments can differ between videos -- collate_pad() below pads them.
    """

    def __init__(self, base_dir, file_label_pairs=None):
        self.base_dir = base_dir
        if file_label_pairs is not None:
            self.files, self.labels, self.label_map = file_label_pairs
        else:
            categories = sorted(
                d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            )
            self.label_map = {cat: i for i, cat in enumerate(categories)}
            self.files, self.labels = [], []
            for cat in categories:
                cat_path = os.path.join(base_dir, cat)
                for f in sorted(os.listdir(cat_path)):
                    if f.endswith(".npy"):
                        self.files.append(os.path.join(cat_path, f))
                        self.labels.append(self.label_map[cat])
            print(f"Classes: {self.label_map}")
            print(f"Total samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])
        if x.ndim == 1:
            x = x[None, :]
        y = self.labels[idx]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

    def subset(self, indices):
        files = [self.files[i] for i in indices]
        labels = [self.labels[i] for i in indices]
        return UCFCrimeDataset(self.base_dir, file_label_pairs=(files, labels, self.label_map))

    def find_normal_index(self):
        for name, idx in self.label_map.items():
            if "normal" in name.lower():
                return idx
        return None


def stratified_split(dataset, val_fraction=0.15, seed=42):
    from sklearn.model_selection import train_test_split
    indices = list(range(len(dataset)))
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            stratify=dataset.labels,
            random_state=seed,
        )
    except ValueError as e:
        print(f"Stratified split failed ({e}); falling back to a random split.")
        train_idx, val_idx = train_test_split(
            indices, test_size=val_fraction, random_state=seed
        )
    return dataset.subset(train_idx), dataset.subset(val_idx)


def class_weights_from_labels(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def collate_pad(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    max_len = padded.size(1)
    pad_mask = torch.arange(max_len)[None, :] >= lengths[:, None]
    labels = torch.stack(labels)
    return padded, lengths, pad_mask, labels


class EarlyStopping:
    def __init__(self, patience=15, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        improved = self.best is None or (
            value > self.best if self.mode == "max" else value < self.best
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def compute_metrics(y_true, y_pred, label_map):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    inv_map = {v: k for k, v in label_map.items()}
    names = [inv_map[i] for i in range(len(label_map))]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "report": classification_report(y_true, y_pred, target_names=names, zero_division=0),
    }


def save_checkpoint(path, model, optimizer, epoch, best_metric, label_map, config):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "label_map": label_map,
        "config": config,
    }, path)


def save_label_map(path, label_map):
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2)