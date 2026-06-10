"""
PyTorch Dataset for AbdomenCT-1K 2D NEPA pretraining.

Loads preprocessed .npy files (one per CT volume, shape [N_slices, 224, 224])
and exposes them as individual 2D slices ready for ViT.

Output per sample:
    pixel_values: torch.Tensor of shape (3, 224, 224), float32
                  - 3-channel (grayscale replicated)
                  - normalized with ImageNet mean/std
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class AbdomenCTSliceDataset(Dataset):
    """
    Each item is one 2D axial slice from a preprocessed volume.
    All volumes share the same preprocessing (window, resize, etc.).

    Args:
        data_dir: folder containing .npy files (one per case)
        train: if True, applies light augmentation (flip)
    """

    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train

        # Build a flat index: list of (case_path, slice_index)
        # This lets us treat the whole dataset as one big slice pool.
        self.index = []
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

        for path in npy_files:
            # Memory-map to read shape without loading the whole volume
            volume = np.load(path, mmap_mode="r")
            num_slices = volume.shape[0]
            for slice_idx in range(num_slices):
                self.index.append((path, slice_idx))

        print(f"AbdomenCTSliceDataset: {len(npy_files)} cases, {len(self.index)} slices")

        # Augmentations applied per-slice
        if train:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.augment = None

        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, slice_idx = self.index[idx]

        # Memory-mapped read: only loads this one slice, not the whole volume
        volume = np.load(path, mmap_mode="r")
        slice_2d = np.array(volume[slice_idx], dtype=np.float32)  # (224, 224) in [0, 1]

        # Convert to 3-channel tensor (C, H, W)
        slice_3ch = np.stack([slice_2d, slice_2d, slice_2d], axis=0)  # (3, 224, 224)
        tensor = torch.from_numpy(slice_3ch)

        # Augment (operates on tensor, expects (C, H, W))
        if self.augment is not None:
            tensor = self.augment(tensor)

        # Normalize with ImageNet stats
        tensor = self.normalize(tensor)

        return {"pixel_values": tensor}


# Quick sanity check you can run after preprocessing
if __name__ == "__main__":
    DATA_DIR = "/content/preprocessed_slices"

    ds = AbdomenCTSliceDataset(DATA_DIR, train=True)
    sample = ds[0]

    print(f"Sample shape: {sample['pixel_values'].shape}")
    print(f"Sample dtype: {sample['pixel_values'].dtype}")
    print(f"Sample range: [{sample['pixel_values'].min():.3f}, "
          f"{sample['pixel_values'].max():.3f}]")
    print(f"Sample mean: {sample['pixel_values'].mean():.3f}")
    print(f"Sample std:  {sample['pixel_values'].std():.3f}")

    # Test a dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    batch = next(iter(loader))
    print(f"\nBatch shape: {batch['pixel_values'].shape}")