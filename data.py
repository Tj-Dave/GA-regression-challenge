import os
import random
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# ------------------------------------------------
# ImageNet transforms (ConvNeXt / ResNet compatible)
# ------------------------------------------------
imagenet_transform = T.Compose([
    T.ToTensor(),                      # Converts (H,W,C) uint8 → (C,H,W) float32 in [0,1]
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# ------------------------------------------------
# Utility functions
# ------------------------------------------------
def load_and_sample_nifti(path, target_frames=16):
    """
    Loads a NIfTI file and returns a float32 numpy array of shape (H, W, 1, T_target).
    """
    img = nib.load(path).get_fdata().astype(np.float32)   # (H, W, 1, T)
    num_frames = img.shape[-1]

    if num_frames >= target_frames:
        indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
        sampled = img[..., indices]
    else:
        repeat_factor = int(np.ceil(target_frames / num_frames))
        repeated = np.tile(img, (1, 1, 1, repeat_factor))
        sampled = repeated[..., :target_frames]

    return sampled  # (H, W, 1, 16)


def preprocess_frame(frame):
    """
    frame: numpy array (H, W, 1)
    Output: torch Tensor (3, H, W) after transforms.
    """
    # Expand to 3 channels
    frame = np.repeat(frame, 3, axis=2)  # (H, W, 3)

    # Normalize to [0, 255] safely
    f_min, f_max = frame.min(), frame.max()
    if f_max - f_min < 1e-6:
        frame = np.zeros_like(frame)
    else:
        frame = (frame - f_min) / (f_max - f_min)

    frame = (frame * 255).astype(np.uint8)

    # Convert to PIL → apply transforms
    frame = Image.fromarray(frame)
    frame = imagenet_transform(frame)
    return frame  # (3, H, W) torch tensor


# ------------------------------------------------
# Training Dataset (Samples use 2 sweeps by default)
# ------------------------------------------------
class SweepDataset(Dataset):
    """
    Training dataset that loads One-TWO random sweeps per sample.
    Output shape: (S=2, T=16, C=3, 224, 224)
    """
    def __init__(self, csv_path, n_sweeps=1, transform=None, load_nifti=True):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.load_nifti = load_nifti
        self.n_sweeps = n_sweeps
        self.sweep_cols = [c for c in self.df.columns if c.startswith('path_nifti')]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        available = row[self.sweep_cols].dropna().tolist()
        if len(available) < self.n_sweeps:
            selected = random.choices(available, k=self.n_sweeps)
        else:
            selected = random.sample(available, self.n_sweeps)

        sweeps_tensor = []

        for path in selected:
            img = load_and_sample_nifti(path)  # (H, W, 1, 16)

            frames = []
            for f in range(img.shape[-1]):
                frame = img[:, :, :, f]  # (H, W, 1)
                frame = preprocess_frame(frame)
                frames.append(frame)

            frames = torch.stack(frames, dim=0)  # (T=16, 3, H, W)
            sweeps_tensor.append(frames)

        sweeps_tensor = torch.stack(sweeps_tensor, dim=0)  # (S, T, 3, H, W)

        label = torch.tensor(row['ga'], dtype=torch.float32)
        return sweeps_tensor, label



# ------------------------------------------------
# Evaluation Dataset (uses ALL sweeps)
# ------------------------------------------------
class SweepEvalDataset(Dataset):
    """
    Validation/testing dataset.
    Loads all sweeps for a study (or first n_sweeps if specified).
    Output shape: (S, T=16, C=3, 224, 224)
    """
    def __init__(self, csv_path, n_sweeps=None, transform=None, load_nifti=True):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.load_nifti = load_nifti
        self.n_sweeps = n_sweeps
        self.sweep_cols = [c for c in self.df.columns if c.startswith('path_nifti')]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sweeps = row[self.sweep_cols].dropna().tolist()

        if self.n_sweeps:
            sweeps = sweeps[:self.n_sweeps]

        sweeps_tensor = []

        for path in sweeps:
            img = load_and_sample_nifti(path)  # (H, W, 1, 16)

            frames = []
            for f in range(img.shape[-1]):
                frame = img[:, :, :, f]
                frame = preprocess_frame(frame)
                frames.append(frame)

            frames = torch.stack(frames, dim=0)  # (T=16, 3, H, W)
            sweeps_tensor.append(frames)

        sweeps_tensor = torch.stack(sweeps_tensor, dim=0)  # (S, T, 3, H, W)

        label = torch.tensor(row['ga'], dtype=torch.float32)
        return sweeps_tensor, label
