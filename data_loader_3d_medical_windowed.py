#!/usr/bin/env python3
"""
Updated data loader for AbdomenCT-1K with CT windowing support.

Replaces simple min-max normalization with clinically-standard windowing.
Usage: Same as original, but add `window=(center, width)` parameter.
"""

import os
from pathlib import Path
from typing import List, Union, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from ct_windowing import apply_ct_window, apply_multi_window


def load_nifti_multiview_slices_windowed(
    path: str,
    window: Optional[Tuple[int, int]] = (40, 400),
    multi_window: Optional[List[Tuple[int, int]]] = None,
) -> List[np.ndarray]:
    """
    Load NIfTI volume with CT windowing.

    Args:
        path: Path to .nii.gz file
        window: (center, width) tuple for windowing. If None, uses min-max normalization
        multi_window: List of (center, width) tuples for multi-channel output

    Returns:
        List of (H, W, 9) or (H, W, C) arrays with multi-view slices
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    vol = nib.load(path).get_fdata()
    if vol.ndim == 4:
        vol = vol[..., 0]

    # Apply windowing or min-max
    if multi_window is not None:
        # Multi-window: stack multiple windows per slice
        vol_windowed = apply_multi_window(vol, windows=multi_window)
    elif window is not None:
        # Single window
        vol = apply_ct_window(vol, center=window[0], width=window[1])
        vol_windowed = vol
    else:
        # Fallback: min-max (original behavior)
        vol = vol - vol.min()
        if vol.max() > 0:
            vol = vol / vol.max()
        vol_windowed = (vol * 255).astype(np.uint8)

    H, W, D = vol_windowed.shape[:3]
    slices = []

    for i in range(D):
        # Axial view: slice along z-axis
        axial = vol_windowed[:, :, i]
        axial_rgb = np.stack([axial, axial, axial], axis=-1)  # (H, W, 3)

        # Sagittal view: slice along x-axis (scaled to match index)
        sag_idx = int(i * (H - 1) / (D - 1)) if D > 1 else 0
        sagittal = vol_windowed[sag_idx, :, :]  # (W, D)
        sagittal = np.transpose(sagittal, (1, 0))  # (D, W)
        sagittal = np.tile(sagittal[:, np.newaxis, :], (1, H // W if H > W else 1, 1))
        sagittal = sagittal.reshape(H, W)
        sagittal_rgb = np.stack([sagittal, sagittal, sagittal], axis=-1)  # (H, W, 3)

        # Coronal view: slice along y-axis (scaled to match index)
        cor_idx = int(i * (W - 1) / (D - 1)) if D > 1 else 0
        coronal = vol_windowed[:, cor_idx, :]  # (H, D)
        coronal = np.tile(coronal[:, np.newaxis, :], (1, W // H if W > H else 1, 1))
        coronal = coronal.reshape(H, W)
        coronal_rgb = np.stack([coronal, coronal, coronal], axis=-1)  # (H, W, 3)

        # Stack all three views: (H, W, 9)
        multiview = np.concatenate([axial_rgb, sagittal_rgb, coronal_rgb], axis=-1)
        slices.append(multiview.astype(np.uint8))

    return slices


class Medical3DDatasetWindowed(Dataset):
    """
    Medical 3D dataset with windowing support.

    Key difference from original: Can apply CT windowing instead of min-max normalization.
    """

    def __init__(
        self,
        root: str,
        transform,
        min_slices: int = 4,
        max_slices: int = None,
        multiview: bool = True,
        window: Optional[Tuple[int, int]] = (40, 400),
        use_windowing: bool = True,
    ):
        """
        Args:
            root: Root folder containing volumes
            transform: torchvision transforms
            min_slices: Minimum slices per volume
            max_slices: Maximum slices per volume
            multiview: If True, output [T, 9, H, W]
            window: (center, width) for CT windowing. If None, uses min-max
            use_windowing: If False, reverts to min-max normalization
        """
        self.transform = transform
        self.min_slices = min_slices
        self.max_slices = max_slices
        self.multiview = multiview
        self.window = window if use_windowing else None
        root = Path(root)

        # Collect volume paths
        nifti_exts = {".nii", ".gz"}
        self.volumes: List[Union[str, Path]] = []

        for p in sorted(root.iterdir()):
            if p.suffix.lower() in nifti_exts or p.name.endswith(".nii.gz"):
                self.volumes.append(str(p))

        if not self.volumes:
            for p in sorted(root.iterdir()):
                if p.is_dir():
                    self.volumes.append(str(p))

        if not self.volumes:
            raise RuntimeError(f"No volumes found in {root}")

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        path = self.volumes[idx]

        # Load with windowing
        if os.path.isdir(path):
            # Folder of slices - use original loader
            from data_loader_3d_medical import load_folder_multiview_slices
            slices = load_folder_multiview_slices(path)
        else:
            # NIfTI file - use windowed loader
            slices = load_nifti_multiview_slices_windowed(path, window=self.window)

        # Optional slice truncation
        if self.max_slices is not None and len(slices) > self.max_slices:
            start = np.random.randint(0, len(slices) - self.max_slices)
            slices = slices[start : start + self.max_slices]

        if len(slices) < self.min_slices:
            raise RuntimeError(f"Volume {path} has only {len(slices)} slices")

        # Process slices
        frames = []
        for s in slices:
            if self.multiview:
                axial_view = s[:, :, :3]
                sagittal_view = s[:, :, 3:6]
                coronal_view = s[:, :, 6:9]

                axial_img = Image.fromarray(axial_view)
                sagittal_img = Image.fromarray(sagittal_view)
                coronal_img = Image.fromarray(coronal_view)

                axial_t = self.transform(axial_img)
                sagittal_t = self.transform(sagittal_img)
                coronal_t = self.transform(coronal_img)

                multiview_t = torch.cat([axial_t, sagittal_t, coronal_t], dim=0)
                frames.append(multiview_t)
            else:
                axial_view = s[:, :, :3]
                img = Image.fromarray(axial_view)
                frames.append(self.transform(img))

        if self.multiview:
            return torch.stack(frames)  # [T, 9, H, W]
        else:
            return torch.stack(frames)  # [T, 3, H, W]


if __name__ == "__main__":
    from data_loader_3d_medical import build_transform, collate_medical_volumes

    # Example: Load with windowing
    transform = build_transform(size=224)

    # With windowing (recommended)
    dataset_windowed = Medical3DDatasetWindowed(
        root="path/to/AbdomenCT-1K/images",
        transform=transform,
        window=(40, 400),  # Soft tissue window
        use_windowing=True,
    )

    # Without windowing (original behavior)
    dataset_original = Medical3DDatasetWindowed(
        root="path/to/AbdomenCT-1K/images",
        transform=transform,
        use_windowing=False,
    )

    dataloader = DataLoader(
        dataset_windowed,
        batch_size=4,
        collate_fn=lambda batch: collate_medical_volumes(batch, multiview=True),
    )
