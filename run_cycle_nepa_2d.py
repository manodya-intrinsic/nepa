"""Single-Step Cycle-NEPA for 2D images — Algorithm 1 (Methodology-1).

Sequence structure for 2D images:
  - One image is split into T patches (T = (image_size / patch_size)^2)
  - z_t  = f_θ(x_t)  : clean patch embedding at position t  (input to transformer)
  - ẑ_{t+1} = H_fwd(Mask(z_{≤t}))  : causal transformer output at t predicts next embedding
  - ẑ_t^recon = H_bwd(ẑ_{t+1})     : backward head reconstructs z_t from the prediction

Losses (Algorithm 1):
  - L_NEPA  = -cos_sim(ẑ_{t+1}, sg[z_{t+1}])   (neg cosine sim, stop-grad on target)
  - L_cycle = ||z_t - ẑ_t^recon||_2^2           (L2 reconstruction, stop-grad on target)
  - L_total = L_NEPA + λ * L_cycle

All three components are updated: θ (patch embedder), H_fwd (causal transformer), H_bwd (BackwardHead).
"""

import argparse
import os
import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize

from datasets import load_dataset, load_from_disk
from PIL import Image

from transformers import AutoImageProcessor

from models.vit_nepa.modeling_vit_nepa import ViTNepaModel
from models.vit_nepa.cycle_heads import BackwardHead


# ── Masking ratios from the paper (section 2.2.1) ────────────────────────────
MASK_RATIOS = [0.75, 0.85, 0.90]


# ── Dataset ───────────────────────────────────────────────────────────────────

class SingleViewDataset(torch.utils.data.Dataset):
    """Returns one augmented view per image. Patches of that view form the sequence."""

    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item.get("image") or item.get("img") or item.get("pixel_values")
        if isinstance(img, dict) and "path" in img:
            img = Image.open(img["path"]).convert("RGB")
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        return self.transform(img.convert("RGB"))


def build_transform(proc: AutoImageProcessor):
    raw_size = getattr(proc, "size", 224)
    size = raw_size.get("shortest_edge", 224) if isinstance(raw_size, dict) else raw_size
    mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
    std  = getattr(proc, "image_std",  [0.229, 0.224, 0.225])
    return Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])


# ── Asymmetric masking ────────────────────────────────────────────────────────

def make_bool_masked_pos(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Random boolean mask: True = masked (replaced with mask token in embeddings).

    Each sample in the batch gets an independent random mask.
    """
    num_masked = int(num_patches * mask_ratio)
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    for i in range(batch_size):
        idx = torch.randperm(num_patches, device=device)[:num_masked]
        mask[i, idx] = True
    return mask


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Single-Step Cycle-NEPA (Algorithm 1)")
    parser.add_argument("--model_id", type=str, default="SixAILab/nepa-base-patch14-224",
                        help="Pretrained NEPA model (use pretrain, not sft).")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--load_from_disk", action="store_true")
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="LR for the BackwardHead.")
    parser.add_argument("--backbone_lr", type=float, default=1e-5,
                        help="LR for the NEPA backbone (lower to preserve pretrained features).")
    parser.add_argument("--cycle_lambda", type=float, default=1.0,
                        help="λ: weight of the cycle consistency loss.")
    parser.add_argument("--mask_ratio", type=float, default=None,
                        help="Fixed masking ratio. If None, sampled randomly from {0.75, 0.85, 0.90}.")
    parser.add_argument("--save_dir", type=str, default="outputs/cycle_nepa")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from.")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ─────────────────────────────────────────────────────────────
    proc  = AutoImageProcessor.from_pretrained(args.model_id)
    model = ViTNepaModel.from_pretrained(args.model_id, trust_remote_code=True)

    # ViTNepaModel defaults to use_mask_token=False.
    # Cycle-NEPA needs it to replace masked patches with a learned mask token.
    if model.embeddings.mask_token is None:
        model.embeddings.mask_token = nn.Parameter(
            torch.zeros(1, 1, model.config.hidden_size)
        )

    model.to(device).train()

    embed_dim   = model.config.hidden_size
    patch_size  = model.config.patch_size
    image_size  = model.config.image_size
    num_patches = (image_size // patch_size) ** 2

    # H_bwd: backward reconstruction head (Algorithm 1, line 7)
    bwd = BackwardHead(embed_dim).to(device)

    # Two LR groups: lower LR for the pretrained backbone to preserve features
    optim = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.backbone_lr, "weight_decay": 0.05},
        {"params": bwd.parameters(),   "lr": args.lr,          "weight_decay": 0.05},
    ])

    start_epoch = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        bwd.load_state_dict(ckpt["bwd_state"])
        optim.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    # ── Dataset ───────────────────────────────────────────────────────────
    if args.train_dir is not None:
        ds = load_dataset(
            "imagefolder",
            data_files={"train": os.path.join(args.train_dir, "**")},
        )["train"]
    elif args.dataset_name is not None:
        if args.load_from_disk:
            raw = load_from_disk(args.dataset_name)
            ds = raw["train"] if "train" in raw else raw
        else:
            ds = load_dataset(args.dataset_name, split="train")
    else:
        raise RuntimeError("Provide --train_dir or --dataset_name")

    dl = DataLoader(
        SingleViewDataset(ds, build_transform(proc)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training loop (Algorithm 1) ────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        bwd.train()
        total_nepa = total_cycle = total = 0.0

        step = 0
        for step, imgs in enumerate(dl):
            imgs = imgs.to(device)
            B = imgs.shape[0]

            # ── Line 4: Asymmetric masking on context z_{≤t} ──────────────
            ratio = args.mask_ratio if args.mask_ratio is not None \
                else random.choice(MASK_RATIOS)
            bool_masked_pos = make_bool_masked_pos(B, num_patches, ratio, device)

            # ── Lines 1-4: Encoder + causal transformer forward pass ───────
            # last_hidden_state : [B, 1+T, D]  position t holds ẑ_{t+1}
            # input_embedding   : [B, 1+T, D]  clean z_t (before masking)
            outputs = model(pixel_values=imgs, bool_masked_pos=bool_masked_pos)

            # Strip CLS token (position 0); work only on the T patch positions
            seq_out = outputs.last_hidden_state[:, 1:, :]  # [B, T, D]  ẑ
            seq_in  = outputs.input_embedding[:, 1:, :]    # [B, T, D]  z (clean)

            # ── Line 5: L_NEPA — negative cosine similarity ───────────────
            # output at position t  →  predicts input at position t+1
            # stop-gradient on the target (sg[z_{t+1}])
            pred   = F.normalize(seq_out[:, :-1, :], dim=-1)         # [B, T-1, D]
            target = F.normalize(seq_in[:, 1:, :].detach(), dim=-1)  # [B, T-1, D]
            loss_nepa = -(pred * target).sum(dim=-1).mean()

            # ── Lines 7-8: L_cycle — backward reconstruction ──────────────
            # H_bwd(ẑ_{t+1}) should recover z_t
            # ẑ_{t+1} at output position t  →  reconstruct z_t at input position t
            z_hat_next = seq_out[:, :-1, :]                  # [B, T-1, D]  ẑ_{t+1}
            recon_t    = bwd(z_hat_next)                     # [B, T-1, D]  ẑ_t^recon
            z_t_target = seq_in[:, :-1, :].detach()          # [B, T-1, D]  sg[z_t]
            loss_cycle = F.mse_loss(recon_t, z_t_target)

            # ── Line 9: Total loss ─────────────────────────────────────────
            loss = loss_nepa + args.cycle_lambda * loss_cycle

            # ── Line 11: Update θ, H_fwd (inside model), H_bwd ───────────
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_nepa  += loss_nepa.item()
            total_cycle += loss_cycle.item()
            total       += loss.item()

            if step % 50 == 0:
                print(
                    f"Epoch {epoch} | Step {step:5d} | "
                    f"loss={loss.item():.4f}  "
                    f"L_nepa={loss_nepa.item():.4f}  "
                    f"L_cycle={loss_cycle.item():.4f}  "
                    f"mask={ratio:.0%}"
                )

        n = step + 1
        print(
            f"\n[Epoch {epoch}]  "
            f"avg_loss={total/n:.4f}  "
            f"avg_L_nepa={total_nepa/n:.4f}  "
            f"avg_L_cycle={total_cycle/n:.4f}\n"
        )

        # Save: model, bwd head, optimizer, and args for full resume
        ckpt_path = os.path.join(args.save_dir, f"cycle_nepa_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "bwd_state":   bwd.state_dict(),
            "optim_state": optim.state_dict(),
            "args":        vars(args),
        }, ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    main()
