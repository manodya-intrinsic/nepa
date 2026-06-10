"""Cycle-NEPA for medical volumetric data (CT / MRI) — Algorithm 1 (Methodology-1).

Architecture
------------
  f_θ   : 2D NEPA encoder (frozen) applied to each slice independently
           → mean-pool of patch tokens  →  one embedding z_t per slice  [D]
  H_fwd : SliceCausalPredictor  — small causal transformer across the slice sequence
           takes  [z_1, ..., z_T]  (with asymmetric masking applied)
           outputs  ẑ_{t+1}  at position t
  H_bwd : BackwardHead MLP
           takes  ẑ_{t+1}  →  reconstructs  z_t

Sequence
--------
  T slices per volume  →  T slice embeddings  →  T-1 predictions / reconstructions

Multi-View Input
----------------
  Each slice position contains 3 orthogonal views (axial, sagittal, coronal) stacked as:
  [T, 9, H, W]  where 9 = 3 planes × 3 RGB channels

Masking
-------
  75-90% of slice embeddings  replaced with a learned mask token BEFORE H_fwd.
  This is inter-slice masking (not patch-level).
  BackwardHead receives the FULL predicted vector (no masking) — asymmetric difficulty.

Losses (Algorithm 1)
--------------------
  L_NEPA   = -cos_sim( ẑ_{t+1},  sg[z_{t+1}] )
  L_cycle  = || z_t  -  H_bwd(ẑ_{t+1}) ||_2^2
  L_total  = L_NEPA  +  λ * L_cycle

Updates
-------
  f_θ is FROZEN — only H_fwd (SliceCausalPredictor) and H_bwd (BackwardHead) are trained.

Data formats supported
----------------------
  1. Folder-of-slices:  one sub-folder per volume, each containing PNG/JPG slice images
                        (files sorted alphabetically = slice order)
  2. NIfTI:             one .nii / .nii.gz file per volume  (requires nibabel)
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from transformers import AutoImageProcessor

from models.vit_nepa.modeling_vit_nepa import ViTNepaModel
from models.vit_nepa.cycle_heads import SliceCausalPredictor, BackwardHead


# ── Masking ratios (paper section 2.2.1) ─────────────────────────────────────
MASK_RATIOS = [0.75, 0.85, 0.90]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_nifti_multiview_slices(path: str) -> List[np.ndarray]:
    """Load a NIfTI volume and return multi-view slices (axial, sagittal, coronal).

    Returns:
        List of shape (H, W, 9) arrays where 9 = [axial_r, axial_g, axial_b,
                                                    sagittal_r, sagittal_g, sagittal_b,
                                                    coronal_r, coronal_g, coronal_b]
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required for NIfTI support: pip install nibabel")
    vol = nib.load(path).get_fdata()          # (H, W, D) or (H, W, D, T)
    if vol.ndim == 4:
        vol = vol[..., 0]                      # take first time point
    # normalise to 0-255
    vol = vol - vol.min()
    if vol.max() > 0:
        vol = vol / vol.max()
    vol = (vol * 255).astype(np.uint8)

    H, W, D = vol.shape
    slices = []

    # Use the maximum dimension as sequence length (axial slices)
    num_slices = D

    for i in range(num_slices):
        # Axial view: slice along z-axis
        axial = vol[:, :, i]
        axial_rgb = np.stack([axial, axial, axial], axis=-1)  # (H, W, 3)

        # Sagittal view: slice along x-axis at corresponding position
        # Map axial index to sagittal index based on volume proportions
        sag_idx = int(i * (H - 1) / (D - 1)) if D > 1 else 0
        sagittal = vol[sag_idx, :, :]  # (W, D)
        # Resize sagittal to match axial spatial dimensions
        sagittal = np.tile(sagittal[:, np.newaxis, :], (1, H // W if H > W else 1, 1))
        sagittal = np.transpose(sagittal, (1, 0, 2))  # -> (H, W, D) orientation
        sagittal = sagittal[:H, :W]  # Crop to H x W
        sagittal_rgb = np.stack([sagittal, sagittal, sagittal], axis=-1)  # (H, W, 3)

        # Coronal view: slice along y-axis at corresponding position
        cor_idx = int(i * (W - 1) / (D - 1)) if D > 1 else 0
        coronal = vol[:, cor_idx, :]  # (H, D)
        # Resize coronal to match axial spatial dimensions
        coronal = np.tile(coronal[:, np.newaxis, :], (1, W // H if W > H else 1, 1))
        coronal = np.transpose(coronal, (0, 2, 1))  # -> (H, D, W) orientation
        coronal = coronal[:H, :W]  # Crop to H x W
        coronal_rgb = np.stack([coronal, coronal, coronal], axis=-1)  # (H, W, 3)

        # Stack all three views: (H, W, 9)
        multiview = np.concatenate([axial_rgb, sagittal_rgb, coronal_rgb], axis=-1)
        slices.append(multiview.astype(np.uint8))

    return slices


def _load_folder_multiview_slices(folder: str) -> List[np.ndarray]:
    """Load multi-view slices from folder.

    Supports two layouts:
      1. Single view (replicated): images directly in folder/
      2. Multi-view (preferred): folder/axial/, folder/sagittal/, folder/coronal/
    """
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    folder_path = Path(folder)

    # Check if multi-view subdirectories exist
    axial_dir = folder_path / "axial"
    sagittal_dir = folder_path / "sagittal"
    coronal_dir = folder_path / "coronal"

    if axial_dir.exists() and sagittal_dir.exists() and coronal_dir.exists():
        # Multi-view layout
        axial_paths = sorted(p for p in axial_dir.iterdir() if p.suffix.lower() in exts)
        sagittal_paths = sorted(p for p in sagittal_dir.iterdir() if p.suffix.lower() in exts)
        coronal_paths = sorted(p for p in coronal_dir.iterdir() if p.suffix.lower() in exts)

        if not (axial_paths and sagittal_paths and coronal_paths):
            raise RuntimeError(f"Multi-view directories exist but missing images in {folder}")

        num_slices = len(axial_paths)
        slices = []

        for i in range(num_slices):
            axial = np.array(Image.open(axial_paths[i]).convert("L"))
            sagittal = np.array(Image.open(sagittal_paths[min(i, len(sagittal_paths) - 1)]).convert("L"))
            coronal = np.array(Image.open(coronal_paths[min(i, len(coronal_paths) - 1)]).convert("L"))

            # Stack: (H, W, 9) where 9 = 3 grayscale converted to RGB x 3 views
            axial_rgb = np.stack([axial, axial, axial], axis=-1)
            sagittal_rgb = np.stack([sagittal, sagittal, sagittal], axis=-1)
            coronal_rgb = np.stack([coronal, coronal, coronal], axis=-1)

            multiview = np.concatenate([axial_rgb, sagittal_rgb, coronal_rgb], axis=-1)
            slices.append(multiview.astype(np.uint8))
    else:
        # Single view (replicate 3x for compatibility)
        paths = sorted(p for p in folder_path.iterdir() if p.suffix.lower() in exts)
        if not paths:
            raise RuntimeError(f"No image files found in {folder}")

        slices = []
        for p in paths:
            img = np.array(Image.open(p).convert("L"))
            img_rgb = np.stack([img, img, img], axis=-1)
            # Replicate view 3x: (H, W, 9)
            multiview = np.concatenate([img_rgb, img_rgb, img_rgb], axis=-1)
            slices.append(multiview.astype(np.uint8))

    return slices


class MedicalVolumeDataset(Dataset):
    """Each item is one volume: a tensor of shape [T, 9, H, W].

    Multi-view slices: 3 orthogonal planes (axial, sagittal, coronal) × 3 RGB channels.

    Supports two layouts:
      folder_of_folders:  root/volume_001/slice_001.png  ...
      nifti:              root/volume_001.nii.gz          ...
    """

    def __init__(self, root: str, transform, min_slices: int = 4, max_slices: int = None):
        self.transform = transform
        self.min_slices = min_slices
        self.max_slices = max_slices
        root = Path(root)

        # collect volume paths
        nifti_exts = {".nii", ".gz"}
        self.volumes: List[Union[str, Path]] = []

        # NIfTI files directly under root
        for p in sorted(root.iterdir()):
            if p.suffix.lower() in nifti_exts or (
                p.name.endswith(".nii.gz")
            ):
                self.volumes.append(str(p))

        # sub-folders of slices
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

        if os.path.isdir(path):
            slices = _load_folder_multiview_slices(path)
        else:
            slices = _load_nifti_multiview_slices(path)

        # optional slice truncation
        if self.max_slices is not None and len(slices) > self.max_slices:
            start = random.randint(0, len(slices) - self.max_slices)
            slices = slices[start: start + self.max_slices]

        if len(slices) < self.min_slices:
            raise RuntimeError(
                f"Volume {path} has only {len(slices)} slices (min={self.min_slices})"
            )

        # process each multi-view slice through transform
        frames = []
        for s in slices:
            # s is shape (H, W, 9): [axial_rgb (0:3), sagittal_rgb (3:6), coronal_rgb (6:9)]
            axial_view = s[:, :, :3]
            sagittal_view = s[:, :, 3:6]
            coronal_view = s[:, :, 6:9]

            # apply transform to each view separately
            axial_img = Image.fromarray(axial_view)
            sagittal_img = Image.fromarray(sagittal_view)
            coronal_img = Image.fromarray(coronal_view)

            axial_t = self.transform(axial_img)          # [3, H, W]
            sagittal_t = self.transform(sagittal_img)    # [3, H, W]
            coronal_t = self.transform(coronal_img)      # [3, H, W]

            # concatenate all three views: [9, H, W]
            multiview_t = torch.cat([axial_t, sagittal_t, coronal_t], dim=0)
            frames.append(multiview_t)

        return torch.stack(frames)  # [T, 9, H, W]


def collate_volumes(batch):
    """Pad volumes to the same number of slices within a batch."""
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
    return torch.stack(padded), torch.stack(masks)      # [B,T,9,H,W], [B,T]


def build_transform(proc: AutoImageProcessor):
    raw_size = getattr(proc, "size", 224)
    size = raw_size.get("shortest_edge", 224) if isinstance(raw_size, dict) else raw_size
    mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
    std  = getattr(proc, "image_std",  [0.229, 0.224, 0.225])
    return Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])


# ── Slice-level masking ───────────────────────────────────────────────────────

def mask_slice_sequence(
    z: torch.Tensor,
    mask_token: torch.Tensor,
    mask_ratio: float,
    valid_mask: torch.Tensor = None,
) -> tuple:
    """Replace mask_ratio fraction of slice embeddings with mask_token.

    Args:
        z:           [B, T, D]  clean slice embeddings
        mask_token:  [1, 1, D]  learnable mask embedding
        mask_ratio:  float      fraction to mask (0.75 / 0.85 / 0.90)
        valid_mask:  [B, T] bool  True = real slice (not padding). If None, all real.

    Returns:
        z_masked:      [B, T, D]
        bool_masked:   [B, T]  bool  True = position was masked
    """
    B, T, D = z.shape
    num_mask = int(T * mask_ratio)
    num_mask = max(1, min(num_mask, T - 1))   # keep at least 1 visible, 1 masked

    bool_masked = torch.zeros(B, T, dtype=torch.bool, device=z.device)
    for i in range(B):
        # only mask real (non-padding) slices
        candidates = torch.where(valid_mask[i])[0] if valid_mask is not None \
            else torch.arange(T, device=z.device)
        n = min(num_mask, len(candidates))
        idx = candidates[torch.randperm(len(candidates), device=z.device)[:n]]
        bool_masked[i, idx] = True

    mask = bool_masked.unsqueeze(-1).to(z.dtype)     # [B, T, 1]
    z_masked = z * (1.0 - mask) + mask_token * mask  # [B, T, D]
    return z_masked, bool_masked


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Medical Cycle-NEPA (Algorithm 1)")
    p.add_argument("--model_id", type=str, default="SixAILab/nepa-base-patch14-224",
                   help="Pretrained 2D NEPA model used as the frozen slice encoder.")
    p.add_argument("--data_root", type=str, required=True,
                   help="Root folder containing one sub-folder (or .nii.gz) per volume.")
    p.add_argument("--max_slices", type=int, default=32,
                   help="Maximum slices per volume (random crop if longer).")
    p.add_argument("--min_slices", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Number of volumes per batch.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cycle_lambda", type=float, default=1.0)
    p.add_argument("--mask_ratio", type=float, default=None,
                   help="Fixed mask ratio. If None, sampled from {0.75, 0.85, 0.90}.")
    p.add_argument("--fwd_layers", type=int, default=2,
                   help="Number of transformer layers in SliceCausalPredictor.")
    p.add_argument("--fwd_heads", type=int, default=8)
    p.add_argument("--save_dir", type=str, default="outputs/cycle_nepa_medical")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── f_θ: frozen 2D NEPA slice encoder ─────────────────────────────────
    proc  = AutoImageProcessor.from_pretrained(args.model_id)
    nepa  = ViTNepaModel.from_pretrained(args.model_id, trust_remote_code=True)
    nepa.to(device).eval()
    for p in nepa.parameters():
        p.requires_grad_(False)

    embed_dim = nepa.config.hidden_size

    # ── Channel projection: 9-channel multi-view → 3-channel for NEPA ──────
    channel_proj = nn.Conv2d(9, 3, kernel_size=1, bias=True).to(device)

    # ── H_fwd: causal predictor across slices ─────────────────────────────
    fwd = SliceCausalPredictor(
        embed_dim  = embed_dim,
        num_heads  = args.fwd_heads,
        num_layers = args.fwd_layers,
        max_slices = args.max_slices + 8,   # small buffer
    ).to(device)

    # ── H_bwd: backward reconstruction head ───────────────────────────────
    bwd = BackwardHead(embed_dim).to(device)

    # ── Learnable mask token for slice-level masking ───────────────────────
    # Shape [1, 1, D] — broadcast over batch and time
    mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim, device=device))
    nn.init.trunc_normal_(mask_token, std=0.02)

    # ── Optimizer: fwd, bwd, channel_proj, and mask_token ─────────────────
    optim = torch.optim.AdamW(
        list(fwd.parameters()) + list(bwd.parameters()) +
        list(channel_proj.parameters()) + [mask_token],
        lr=args.lr,
        weight_decay=0.05,
    )

    start_epoch = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        fwd.load_state_dict(ckpt["fwd_state"])
        bwd.load_state_dict(ckpt["bwd_state"])
        channel_proj.load_state_dict(ckpt["channel_proj_state"])
        mask_token.data.copy_(ckpt["mask_token"])
        optim.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    # ── Dataset ────────────────────────────────────────────────────────────
    transform = build_transform(proc)
    dataset = MedicalVolumeDataset(
        root       = args.data_root,
        transform  = transform,
        min_slices = args.min_slices,
        max_slices = args.max_slices,
    )
    dl = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collate_volumes,
        pin_memory  = (device == "cuda"),
        drop_last   = True,
    )
    print(f"Dataset: {len(dataset)} volumes  |  {len(dl)} batches per epoch")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        fwd.train()
        bwd.train()
        total_nepa = total_cycle = total = 0.0
        step = 0

        for step, (volumes, valid_mask) in enumerate(dl):
            # volumes:    [B, T, 9, H, W]  (multi-view: 3 planes × 3 RGB)
            # valid_mask: [B, T]  True = real slice
            volumes    = volumes.to(device)
            valid_mask = valid_mask.to(device)
            B, T, C, H, W = volumes.shape

            # ── Step 1: extract one embedding per slice with frozen NEPA ──
            # flatten batch × time → process all slices in parallel
            slices = volumes.view(B * T, C, H, W)

            # Project 9 channels → 3 channels
            slices = channel_proj(slices)  # [B*T, 3, H, W]

            with torch.no_grad():
                out = nepa(pixel_values=slices)
                # mean pool patch tokens (NOT CLS — useless in causal NEPA)
                # last_hidden_state: [B*T, 1+num_patches, D]
                z_flat = out.last_hidden_state[:, 1:, :].mean(dim=1)  # [B*T, D]

            z = z_flat.view(B, T, embed_dim)   # [B, T, D]  clean slice embeddings

            # ── Step 2: inter-slice asymmetric masking ─────────────────────
            ratio = args.mask_ratio if args.mask_ratio is not None \
                else random.choice(MASK_RATIOS)
            z_masked, _ = mask_slice_sequence(z, mask_token, ratio, valid_mask)
            # z_masked: [B, T, D]  — 75-90% of slice positions are mask tokens

            # ── Step 3: H_fwd — causal prediction across slice sequence ────
            z_hat = fwd(z_masked)              # [B, T, D]  ẑ_{t+1} at position t

            # ── Step 4: L_NEPA — negative cosine similarity ───────────────
            # output at position t  predicts  ground-truth at position t+1
            pred   = F.normalize(z_hat[:, :-1, :], dim=-1)         # [B, T-1, D]
            target = F.normalize(z[:, 1:, :].detach(), dim=-1)     # [B, T-1, D]
            loss_nepa = -(pred * target).sum(dim=-1).mean()

            # ── Step 5: L_cycle — backward reconstruction ─────────────────
            # H_bwd(ẑ_{t+1}) should recover z_t
            recon_t    = bwd(z_hat[:, :-1, :])                     # [B, T-1, D]
            z_t_target = z[:, :-1, :].detach()                     # [B, T-1, D]
            loss_cycle = F.mse_loss(recon_t, z_t_target)

            # ── Step 6: total loss ─────────────────────────────────────────
            loss = loss_nepa + args.cycle_lambda * loss_cycle

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_nepa  += loss_nepa.item()
            total_cycle += loss_cycle.item()
            total       += loss.item()

            if step % 20 == 0:
                print(
                    f"Epoch {epoch} | Step {step:4d} | "
                    f"loss={loss.item():.4f}  "
                    f"L_nepa={loss_nepa.item():.4f}  "
                    f"L_cycle={loss_cycle.item():.4f}  "
                    f"mask={ratio:.0%}  T={T}"
                )

        n = step + 1
        print(
            f"\n[Epoch {epoch}]  "
            f"avg_loss={total/n:.4f}  "
            f"avg_L_nepa={total_nepa/n:.4f}  "
            f"avg_L_cycle={total_cycle/n:.4f}\n"
        )

        ckpt_path = os.path.join(args.save_dir, f"cycle_nepa_medical_epoch{epoch}.pt")
        torch.save({
            "epoch":                epoch,
            "fwd_state":            fwd.state_dict(),
            "bwd_state":            bwd.state_dict(),
            "channel_proj_state":   channel_proj.state_dict(),
            "mask_token":           mask_token.data,
            "optim_state":          optim.state_dict(),
            "args":                 vars(args),
        }, ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    main()
