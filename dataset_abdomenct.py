"""
PyTorch Dataset for AbdomenCT-1K 2D NEPA pretraining.

UPDATED: now generates random patch masks (75% masked) to prevent
representation collapse on visually-uniform CT slices.

Output per sample:
    pixel_values:    (3, 224, 224) float32
    bool_masked_pos: (256,) bool   - True = this patch is masked
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# NEPA masking config
IMAGE_SIZE  = 224
PATCH_SIZE  = 14
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 256
MASK_RATIO  = 0.75


def random_masking(num_patches, mask_ratio):
    """Pick a random subset of patch positions to mask."""
    num_masked = int(num_patches * mask_ratio)
    indices = torch.randperm(num_patches)[:num_masked]
    mask = torch.zeros(num_patches, dtype=torch.bool)
    mask[indices] = True
    return mask


class AbdomenCTSliceDataset(Dataset):
    """
    Args:
        data_dir: folder containing .npy files (one per case)
        train:    if True, applies augmentation (flip) and random masking.
                  if False, no augmentation, no masking (used for clean eval).
        mask_ratio: fraction of patches to mask during training (default 0.75)
    """

    def __init__(self, data_dir, train=True, mask_ratio=MASK_RATIO):
        self.data_dir = data_dir
        self.train = train
        self.mask_ratio = mask_ratio

        # Flat index: (case_path, slice_index)
        self.index = []
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

        if len(npy_files) == 0:
            raise FileNotFoundError(
                f"No .npy files found in {data_dir}. "
                "Did you run preprocess_abdomenct.py first?"
            )

        for path in npy_files:
            volume = np.load(path, mmap_mode="r")
            for slice_idx in range(volume.shape[0]):
                self.index.append((path, slice_idx))

        print(
            f"AbdomenCTSliceDataset: {len(npy_files)} cases, "
            f"{len(self.index)} slices, train={train}, mask_ratio={mask_ratio}"
        )

        if train:
            self.augment = transforms.RandomHorizontalFlip(p=0.5)
        else:
            self.augment = None

        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, slice_idx = self.index[idx]

        volume = np.load(path, mmap_mode="r")
        slice_2d = np.array(volume[slice_idx], dtype=np.float32)

        # 3-channel
        slice_3ch = np.stack([slice_2d, slice_2d, slice_2d], axis=0)
        tensor = torch.from_numpy(slice_3ch)

        if self.augment is not None:
            tensor = self.augment(tensor)

        tensor = self.normalize(tensor)

        # Generate per-slice random mask only during training
        if self.train and self.mask_ratio > 0:
            bool_masked_pos = random_masking(NUM_PATCHES, self.mask_ratio)
        else:
            bool_masked_pos = torch.zeros(NUM_PATCHES, dtype=torch.bool)

        return {
            "pixel_values": tensor,
            "bool_masked_pos": bool_masked_pos,
        }


if __name__ == "__main__":
    DATA_DIR = "/content/preprocessed_slices"

    ds = AbdomenCTSliceDataset(DATA_DIR, train=True)
    sample = ds[0]

    print(f"pixel_values shape: {sample['pixel_values'].shape}")
    print(f"bool_masked_pos shape: {sample['bool_masked_pos'].shape}")
    print(f"Masked positions: {sample['bool_masked_pos'].sum().item()}/256")