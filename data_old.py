import os
import random
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# ----------------------------
# ImageNet transforms
# ----------------------------
imagenet_transform = T.Compose([
    T.ToTensor(),  # H, W, C -> C, H, W
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# ----------------------------
# Training dataset: single sweep per sample
# ----------------------------
class SweepDataset(Dataset):
    """
    Dataset class for training.
    Each sample contains a single sweep (randomly chosen from available sweeps).
    """
    def __init__(self, csv_path, transform=None, load_nifti=True):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.load_nifti = load_nifti
        self.sweep_cols = [c for c in self.df.columns if c.startswith('path_nifti')]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = random.choice(row[self.sweep_cols])

        # Load NIfTI file and preprocess
        if self.load_nifti:
            img = nib.load(path).get_fdata().astype(np.float32)
            num_frames = img.shape[-1]
            target_frames = 16

            if num_frames >= target_frames:
                # Uniformly sample 16 frames
                indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
                sampled_img = img[..., indices]
            else:
                # Repeat frames if fewer than 16
                repeat_factor = int(np.ceil(target_frames / num_frames))
                repeated_img = np.tile(img, (1, 1, 1, repeat_factor))
                sampled_img = repeated_img[..., :target_frames]
            img = sampled_img
        else:
            img = path

        # Apply transforms to each frame
        frames = []
        for f in range(img.shape[-1]):
            frame = np.repeat(img[:, :, :, f], 3, axis=2)  # 3 channels
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        frames = torch.stack(frames, dim=0)  # (T, C, H, W)

        label = torch.tensor(row['ga'], dtype=torch.float32)
        return frames, label


# ----------------------------
# Validation/Test dataset: multiple sweeps per sample
# ----------------------------
class SweepEvalDataset(Dataset):
    """
    Dataset class for validation and testing.
    Each sample contains multiple sweeps per study.
    """
    def __init__(self, csv_path, n_sweeps=None, transform=None, load_nifti=True):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.load_nifti = load_nifti
        self.sweep_cols = [c for c in self.df.columns if c.startswith('path_nifti')]
        self.n_sweeps = n_sweeps

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sweeps = row[self.sweep_cols].tolist()
        if self.n_sweeps:
            sweeps = sweeps[:self.n_sweeps]

        all_sweeps = []
        for path in sweeps:
            if self.load_nifti:
                img = nib.load(path).get_fdata().astype(np.float32)
                target_frames = 16
                num_frames = img.shape[-1]

                if num_frames >= target_frames:
                    indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
                    sampled_img = img[..., indices]
                else:
                    repeat_factor = int(np.ceil(target_frames / num_frames))
                    repeated_img = np.tile(img, (1, 1, 1, repeat_factor))
                    sampled_img = repeated_img[..., :target_frames]
                img = sampled_img

                # Apply transforms
                frames = []
                for f in range(img.shape[-1]):
                    frame = np.repeat(img[:, :, :, f], 3, axis=2)
                    if self.transform:
                        frame = self.transform(frame)
                    frames.append(frame)
                frames = torch.stack(frames, dim=0)  # (T, C, H, W)
            else:
                frames = path

            all_sweeps.append(frames)

        all_sweeps = torch.stack(all_sweeps, dim=0)  # (num_sweeps, T, C, H, W)
        label = torch.tensor(row['ga'], dtype=torch.float32)
        return all_sweeps, label

