"""
Dataset for slice-sequence NEPA pretraining on AbdomenCT-1K.

Reads the preprocessed .npy files produced by preprocess_abdomenct.py
(z-axis resampled to TARGET_SPACING_Z_MM = 2.5mm).

Each __getitem__ returns ONE stack of N consecutive slices from one case.
The model treats the stack as a 3D volume input: (3, N, H, W).

Uses data_split.json to know which cases belong to which split.

Output format per sample:
    pixel_values: torch.FloatTensor of shape (3, num_slices_per_sample, 224, 224)
                  3 channels (grayscale slice replicated 3x), num_slices stacked.
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class AbdomenCTSliceSequenceDataset(Dataset):
    """Returns stacks of N consecutive slices for slice-sequence NEPA.

    Args:
        image_dir: folder with the per-case .npy files (preprocessed, z-resampled)
        split_json: path to data_split.json
        split: one of "train", "val", "test"
        num_slices_per_sample: how many consecutive slices per sample (default 8)
        samples_per_case_per_epoch: how many random N-slice windows to draw from
            each case per epoch. Bigger = more data per epoch but more redundancy.
        train: if True, sample random windows; if False, deterministic windows
            (used for val/test).
        replicate_channels: if True, replicate the grayscale slice to 3 channels
            (default True — matches the ViT's expected num_channels=3).
    """

    def __init__(
        self,
        image_dir: str,
        split_json: str,
        split: str = "train",
        num_slices_per_sample: int = 8,
        samples_per_case_per_epoch: int = 4,
        train: bool = True,
        replicate_channels: bool = True,
    ):
        super().__init__()

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.image_dir = image_dir
        self.num_slices_per_sample = num_slices_per_sample
        self.train = train
        self.replicate_channels = replicate_channels

        # Load the case-level split
        with open(split_json) as f:
            split_data = json.load(f)
        case_ids = split_data[split]

        # Build the index: list of (case_path, n_slices_in_case) for cases that
        # have at least num_slices_per_sample slices.
        self.cases = []
        skipped = 0
        for case_id in case_ids:
            path = os.path.join(image_dir, f"{case_id}.npy")
            if not os.path.exists(path):
                skipped += 1
                continue
            n = np.load(path, mmap_mode="r").shape[0]
            if n < num_slices_per_sample:
                # Skip cases too short to form even one N-slice window
                skipped += 1
                continue
            self.cases.append((path, n))

        if not self.cases:
            raise RuntimeError(
                f"No usable cases found for split={split!r}. "
                f"image_dir={image_dir}, split_json={split_json}. "
                f"Check that preprocessing produced .npy files and the split JSON "
                f"references the same case IDs."
            )

        # For train: each case contributes K random windows per epoch
        # For val/test: each case contributes one fixed window (deterministic)
        if train:
            self.samples_per_case = samples_per_case_per_epoch
        else:
            self.samples_per_case = 1

        print(
            f"AbdomenCTSliceSequenceDataset[{split}]: "
            f"{len(self.cases)} cases, {self.samples_per_case} sample(s) per case, "
            f"{num_slices_per_sample} slices each. "
            f"Total samples per epoch: {len(self)}. "
            f"Skipped: {skipped}."
        )

    def __len__(self):
        return len(self.cases) * self.samples_per_case

    def __getitem__(self, idx):
        case_idx = idx // self.samples_per_case
        path, n = self.cases[case_idx]

        if self.train:
            # Random window
            max_start = n - self.num_slices_per_sample
            start = random.randint(0, max_start)
        else:
            # Deterministic: middle of the case
            start = (n - self.num_slices_per_sample) // 2

        # mmap_mode='r' avoids loading the whole volume into memory
        vol = np.load(path, mmap_mode="r")
        stack = vol[start : start + self.num_slices_per_sample]   # (N, H, W) float16

        # Convert to float32 tensor
        stack = torch.from_numpy(stack.astype(np.float32))         # (N, H, W)

        # Add channel dimension
        if self.replicate_channels:
            # (N, H, W) -> (3, N, H, W) — replicate the grayscale to 3 channels
            stack = stack.unsqueeze(0).repeat(3, 1, 1, 1)
        else:
            # (N, H, W) -> (1, N, H, W)
            stack = stack.unsqueeze(0)

        return {"pixel_values": stack}