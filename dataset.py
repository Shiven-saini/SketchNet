"""
dataset.py — QuickDraw Bitmap Dataset for CNN training

Loads .npy files (28×28 uint8 arrays, white drawing on black bg).
Upscales to 64×64, applies augmentation, returns normalized tensors.

Quick Draw .npy format:
    shape : (N, 784)  — N drawings, each flattened 28×28
    dtype : uint8
    values: 0 = background (black), 255 = stroke (white)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


# ── Augmentation ──────────────────────────────────────────────────────────────

class SketchAugment:
    """
    Augmentations that make sense for hand-drawn sketches.
    NO color jitter, NO random erasing — those hurt sketch models.
    DO: rotation, scale, translate, horizontal flip (selective), blur.
    """
    def __init__(self, train=True):
        self.train = train

    def __call__(self, img):
        # img: PIL Image or tensor (C, H, W) float [0,1]
        if not self.train:
            return img

        # Random rotation ±15°
        if random.random() > 0.3:
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle, fill=0)

        # Random affine: translate + scale
        if random.random() > 0.3:
            img = TF.affine(
                img,
                angle=0,
                translate=[random.randint(-6, 6), random.randint(-6, 6)],
                scale=random.uniform(0.85, 1.15),
                shear=random.uniform(-5, 5),
                fill=0,
            )

        # Horizontal flip — only for symmetric categories
        # (clock, sun, moon, star, flower, pizza are symmetric; cat/dog etc. are too)
        if random.random() > 0.5:
            img = TF.hflip(img)

        # Occasional Gaussian blur (simulates thick/thin strokes)
        if random.random() > 0.85:
            img = TF.gaussian_blur(img, kernel_size=3, sigma=0.8)

        return img


# ── Dataset ───────────────────────────────────────────────────────────────────

class QuickDrawCNNDataset(Dataset):
    """
    Args:
        data_dir      : folder with .npy files
        categories    : list of class names (must match filenames)
        img_size      : resize to this (default 64 — good balance of detail/speed)
        max_per_class : cap per category (None = use all ~100k)
        split         : 'train', 'val', 'test'
        split_ratios  : (train, val, test)
    """

    def __init__(
        self,
        data_dir,
        categories,
        img_size=64,
        max_per_class=50_000,
        split="train",
        split_ratios=(0.8, 0.1, 0.1),
    ):
        self.img_size = img_size
        self.categories = categories
        self.label2idx = {c: i for i, c in enumerate(categories)}
        self.is_train = (split == "train")

        train_r, val_r, _ = split_ratios

        all_imgs   = []
        all_labels = []

        bar = tqdm(
            categories,
            desc=f"Loading [{split:5s}]",
            unit="class",
            dynamic_ncols=True,
            colour="cyan",
        )
        total_samples = 0

        for cat in bar:
            path = Path(data_dir) / f"{cat}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing: {path}\nRun: bash download_dataset.sh"
                )

            file_mb = path.stat().st_size / 1e6
            bar.set_postfix(cls=cat, size=f"{file_mb:.1f}MB", samples=f"{total_samples:,}")

            data = np.load(path)          # (N, 784) uint8
            N = len(data)

            # deterministic split
            train_end = int(N * train_r)
            val_end   = int(N * (train_r + val_r))

            if split == "train":
                data = data[:train_end]
            elif split == "val":
                data = data[train_end:val_end]
            else:
                data = data[val_end:]

            if max_per_class:
                data = data[:max_per_class]

            label = self.label2idx[cat]
            all_imgs.append(data)
            all_labels.extend([label] * len(data))
            total_samples += len(data)
            bar.set_postfix(cls=cat, size=f"{file_mb:.1f}MB", samples=f"{total_samples:,}")

        bar.close()

        print(f"  Concatenating {len(all_imgs)} arrays...", end=" ", flush=True)
        self.data   = np.concatenate(all_imgs, axis=0)   # (total, 784)
        self.labels = np.array(all_labels, dtype=np.int64)
        ram_mb = self.data.nbytes / 1e6
        print(f"done  ({ram_mb:.0f} MB in RAM)")

        # Augmentation pipeline
        self.augment = SketchAugment(train=self.is_train)

        # Base transform (resize + normalize)
        # QuickDraw is white-on-black; we keep it that way
        self.base_tf = T.Compose([
            T.Resize((img_size, img_size), antialias=True),
        ])

        print(f"[{split:5s}] {len(self.labels):>8,} samples | "
              f"{len(categories)} classes | {img_size}×{img_size}px")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load and reshape
        img = self.data[idx].reshape(28, 28).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).unsqueeze(0)   # (1, 28, 28)

        # Resize to target size
        img_t = self.base_tf(img_t)                  # (1, 64, 64)

        # Augment
        img_t = self.augment(img_t)

        # Normalize to [-1, 1]  (mean=0.5, std=0.5 for sketch data)
        img_t = img_t * 2.0 - 1.0

        return img_t, torch.tensor(self.labels[idx])


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(data_dir, categories, img_size=64, max_per_class=50_000,
                    batch_size=256, num_workers=4):
    common = dict(
        data_dir=data_dir,
        categories=categories,
        img_size=img_size,
        max_per_class=max_per_class,
    )
    train_ds = QuickDrawCNNDataset(split="train", **common)
    val_ds   = QuickDrawCNNDataset(split="val",   **common)
    test_ds  = QuickDrawCNNDataset(split="test",  **common)

    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, train_ds.label2idx
