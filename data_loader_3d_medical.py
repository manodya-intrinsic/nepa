"""Data loader for 3D medical imaging (AbdomenCT-1K / CHAOS) → 2D slices.

Converts 3D volumes into multi-view 2D slices (axial, sagittal, coronal).
"""

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


# ── Load 3D volumes and extract multi-view slices ──────────────────────────────

def load_nifti_multiview_slices(path: str) -> List[np.ndarray]:
    """Load NIfTI volume and extract multi-view slices.

    Returns:
        List of (H, W, 9) arrays: [axial_rgb (0:3), sagittal_rgb (3:6), coronal_rgb (6:9)]
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    vol = nib.load(path).get_fdata()  # (H, W, D) or (H, W, D, T)
    if vol.ndim == 4:
        vol = vol[..., 0]  # take first time point

    # normalize to 0-255
    vol = vol - vol.min()
    if vol.max() > 0:
        vol = vol / vol.max()
    vol = (vol * 255).astype(np.uint8)

    H, W, D = vol.shape
    slices = []

    for i in range(D):
        # Axial view: slice along z-axis
        axial = vol[:, :, i]
        axial_rgb = np.stack([axial, axial, axial], axis=-1)  # (H, W, 3)

        # Sagittal view: slice along x-axis (scaled to match index)
        sag_idx = int(i * (H - 1) / (D - 1)) if D > 1 else 0
        sagittal = vol[sag_idx, :, :]  # (W, D)
        sagittal = np.transpose(sagittal, (1, 0))  # (D, W)
        # Resize to H x W
        sagittal = np.tile(sagittal[:, np.newaxis, :], (1, H // W if H > W else 1, 1))
        sagittal = sagittal.reshape(H, W)
        sagittal_rgb = np.stack([sagittal, sagittal, sagittal], axis=-1)  # (H, W, 3)

        # Coronal view: slice along y-axis (scaled to match index)
        cor_idx = int(i * (W - 1) / (D - 1)) if D > 1 else 0
        coronal = vol[:, cor_idx, :]  # (H, D)
        # Resize to H x W
        coronal = np.tile(coronal[:, np.newaxis, :], (1, W // H if W > H else 1, 1))
        coronal = coronal.reshape(H, W)
        coronal_rgb = np.stack([coronal, coronal, coronal], axis=-1)  # (H, W, 3)

        # Stack all three views: (H, W, 9)
        multiview = np.concatenate([axial_rgb, sagittal_rgb, coronal_rgb], axis=-1)
        slices.append(multiview.astype(np.uint8))

    return slices


def load_folder_multiview_slices(folder: str) -> List[np.ndarray]:
    """Load multi-view slices from folder.

    Supports:
      - Multi-view: folder/axial/, folder/sagittal/, folder/coronal/
      - Single-view: images directly in folder (replicated to 3 views)
    """
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    folder_path = Path(folder)

    # Check for multi-view subdirectories
    axial_dir = folder_path / "axial"
    sagittal_dir = folder_path / "sagittal"
    coronal_dir = folder_path / "coronal"

    if axial_dir.exists() and sagittal_dir.exists() and coronal_dir.exists():
        # Multi-view layout
        axial_paths = sorted(p for p in axial_dir.iterdir() if p.suffix.lower() in exts)
        sagittal_paths = sorted(p for p in sagittal_dir.iterdir() if p.suffix.lower() in exts)
        coronal_paths = sorted(p for p in coronal_dir.iterdir() if p.suffix.lower() in exts)

        if not (axial_paths and sagittal_paths and coronal_paths):
            raise RuntimeError(f"Multi-view dirs exist but missing images in {folder}")

        slices = []
        for i in range(len(axial_paths)):
            axial = np.array(Image.open(axial_paths[i]).convert("L"))
            sagittal = np.array(Image.open(sagittal_paths[min(i, len(sagittal_paths) - 1)]).convert("L"))
            coronal = np.array(Image.open(coronal_paths[min(i, len(coronal_paths) - 1)]).convert("L"))

            axial_rgb = np.stack([axial, axial, axial], axis=-1)
            sagittal_rgb = np.stack([sagittal, sagittal, sagittal], axis=-1)
            coronal_rgb = np.stack([coronal, coronal, coronal], axis=-1)

            multiview = np.concatenate([axial_rgb, sagittal_rgb, coronal_rgb], axis=-1)
            slices.append(multiview.astype(np.uint8))
    else:
        # Single-view (replicate to 3 views)
        paths = sorted(p for p in folder_path.iterdir() if p.suffix.lower() in exts)
        if not paths:
            raise RuntimeError(f"No images found in {folder}")

        slices = []
        for p in paths:
            img = np.array(Image.open(p).convert("L"))
            img_rgb = np.stack([img, img, img], axis=-1)
            # Replicate 3x: (H, W, 9)
            multiview = np.concatenate([img_rgb, img_rgb, img_rgb], axis=-1)
            slices.append(multiview.astype(np.uint8))

    return slices


# ── Dataset class ─────────────────────────────────────────────────────────────

class Medical3DDataset(Dataset):
    """Converts 3D medical volumes into 2D slices for 2D model training.

    Outputs:
        - Single view: [T, 3, H, W] where T = number of slices
        - Multi-view: [T, 9, H, W] where 9 = 3 planes × 3 channels
    """

    def __init__(
        self,
        root: str,
        transform,
        min_slices: int = 4,
        max_slices: int = None,
        multiview: bool = True,
    ):
        """
        Args:
            root: Root folder containing volumes (NIfTI files or subfolders)
            transform: torchvision transforms for preprocessing
            min_slices: Minimum slices per volume
            max_slices: Maximum slices per volume (random crop if longer)
            multiview: If True, output [T, 9, H, W]. If False, [T, 3, H, W]
        """
        self.transform = transform
        self.min_slices = min_slices
        self.max_slices = max_slices
        self.multiview = multiview
        root = Path(root)

        # Collect volume paths
        nifti_exts = {".nii", ".gz"}
        self.volumes: List[Union[str, Path]] = []

        # NIfTI files directly under root
        for p in sorted(root.iterdir()):
            if p.suffix.lower() in nifti_exts or p.name.endswith(".nii.gz"):
                self.volumes.append(str(p))

        # Subfolders of slices
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

        # Load multi-view slices
        if os.path.isdir(path):
            slices = load_folder_multiview_slices(path)
        else:
            slices = load_nifti_multiview_slices(path)

        # Optional slice truncation
        if self.max_slices is not None and len(slices) > self.max_slices:
            start = np.random.randint(0, len(slices) - self.max_slices)
            slices = slices[start : start + self.max_slices]

        if len(slices) < self.min_slices:
            raise RuntimeError(
                f"Volume {path} has only {len(slices)} slices (min={self.min_slices})"
            )

        # Process slices through transform
        frames = []
        for s in slices:
            if self.multiview:
                # Extract and transform each view: (H, W, 9) → 3 RGB views
                axial_view = s[:, :, :3]
                sagittal_view = s[:, :, 3:6]
                coronal_view = s[:, :, 6:9]

                axial_img = Image.fromarray(axial_view)
                sagittal_img = Image.fromarray(sagittal_view)
                coronal_img = Image.fromarray(coronal_view)

                axial_t = self.transform(axial_img)  # [3, H, W]
                sagittal_t = self.transform(sagittal_img)
                coronal_t = self.transform(coronal_img)

                # Stack all views: [9, H, W]
                multiview_t = torch.cat([axial_t, sagittal_t, coronal_t], dim=0)
                frames.append(multiview_t)
            else:
                # Use only axial view: (H, W, 3)
                axial_view = s[:, :, :3]
                img = Image.fromarray(axial_view)
                frames.append(self.transform(img))  # [3, H, W]

        if self.multiview:
            return torch.stack(frames)  # [T, 9, H, W]
        else:
            return torch.stack(frames)  # [T, 3, H, W]


# ── Collate function for variable-length sequences ──────────────────────────────

def collate_medical_volumes(batch, multiview=True):
    """Pad volumes to same sequence length within batch."""
    max_t = max(v.shape[0] for v in batch)
    padded, masks = [], []

    for v in batch:
        t = v.shape[0]
        pad = torch.zeros(max_t - t, *v.shape[1:], dtype=v.dtype)
        padded.append(torch.cat([v, pad], dim=0))

        # True = real slice, False = padding
        m = torch.zeros(max_t, dtype=torch.bool)
        m[:t] = True
        masks.append(m)

    output_shape = "[B, T, 9, H, W]" if multiview else "[B, T, 3, H, W]"
    return torch.stack(padded), torch.stack(masks), output_shape


# ── Build transforms ──────────────────────────────────────────────────────────

def build_transform(size: int = 224):
    """Create standard ImageNet-style transform."""
    return Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ── Example usage ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: Load AbdomenCT-1K
    transform = build_transform(size=224)

    dataset = Medical3DDataset(
        root="path/to/AbdomenCT-1K",
        transform=transform,
        min_slices=4,
        max_slices=32,
        multiview=True,  # Use multi-view slices
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda batch: collate_medical_volumes(batch, multiview=True),
        pin_memory=True,
        drop_last=True,
    )

    # Iterate
    for batch_idx, (volumes, valid_mask, output_shape) in enumerate(dataloader):
        print(f"Batch {batch_idx}: {output_shape}")
        print(f"  Volumes shape: {volumes.shape}")
        print(f"  Valid mask shape: {valid_mask.shape}")
        if batch_idx == 0:
            break
