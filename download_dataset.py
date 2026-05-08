#!/usr/bin/env python3
"""
Download datasets from Hugging Face Hub using small presets.

This script keeps dataset-specific settings in one place so switching to a
different dataset later usually means changing only the `--dataset` argument
or updating one preset entry.

Usage:
    python download_dataset.py [--local_dir /path/to/save]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


# ============================================================================
# LEGACY CODE (COMMENTED OUT)
# ============================================================================
# This was the original code for downloading ImageNet-1k from Hugging Face.
# Kept for reference. Use the HMDB51 download functions below instead.
# 
# from datasets import load_dataset
# 
# dataset = load_dataset("ILSVRC/imagenet-1k", revision="4603483700ee984ea9debe3ddbfdeae86f6489eb", trust_remote_code=True)
# 
# dataset.save_to_disk("data/imagenet-1k-hf")
# ============================================================================


DATASET_PRESETS = {
    "hmdb51": {
        "repo_id": "jili5044/hmdb51",
        "repo_type": "dataset",
        "default_local_dir": "./data/hmdb51_local",
        "expected_extension": ".avi",
        "description": "HMDB51 video action recognition dataset",
    },
}


def normalize_extension(extension: Optional[str]) -> Optional[str]:
    if not extension:
        return None
    return extension if extension.startswith(".") else f".{extension}"


def download_dataset(repo_id: str, local_dir: str, repo_type: str = "dataset") -> str:
    """
    Download a dataset from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face Hub dataset ID.
        local_dir: Local directory to save the dataset.
        repo_type: Hugging Face repo type. Defaults to "dataset".
    
    Returns:
        Path to the downloaded dataset.
    
    Raises:
        ImportError: If huggingface_hub is not installed.
        Exception: If download fails.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)

    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    print(f"[INFO] Target directory: {local_dir}")
    print(f"[INFO] Repository: {repo_id} ({repo_type})")

    # Download
    print("[INFO] Downloading dataset from Hugging Face Hub...")
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
        )
        print(f"[INFO] Download complete: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)


def verify_structure(local_dir: str, expected_extension: Optional[str] = None) -> bool:
    """
    Verify the downloaded dataset structure and list top-level folders.
    
    Args:
        local_dir: Path to the downloaded dataset.
        expected_extension: Optional file extension expected in the dataset,
            for example ".avi" for HMDB51.
    
    Returns:
        True if structure is valid, False otherwise.
    """
    local_path = Path(local_dir)
    
    if not local_path.exists():
        print(f"[ERROR] Directory does not exist: {local_dir}")
        return False

    # Find all class folders (assume they are direct subdirectories)
    class_folders = sorted([d for d in local_path.iterdir() if d.is_dir() and not d.name.startswith(".")])
    
    if not class_folders:
        print(f"[WARNING] No top-level folders found in {local_dir}")

    if class_folders:
        print(f"[INFO] Found {len(class_folders)} top-level folders")
    
    # Print first 5 folders with a file count summary.
    if class_folders:
        print("[INFO] First 5 folders:")
        for i, folder in enumerate(class_folders[:5]):
            if expected_extension:
                file_count = len(list(folder.rglob(f"*{expected_extension}")))
                print(f"  {i+1}. {folder.name}/ ({file_count} {expected_extension} files)")
            else:
                file_count = len([path for path in folder.rglob("*") if path.is_file()])
                print(f"  {i+1}. {folder.name}/ ({file_count} files)")

    # Check if the expected files exist.
    all_expected = list(local_path.rglob(f"*{expected_extension}")) if expected_extension else []
    all_zip = list(local_path.rglob("*.zip"))
    
    if all_zip:
        print(f"[WARNING] Found {len(all_zip)} .zip files")
        return False
    
    if expected_extension:
        if all_expected:
            print(f"[INFO] Verified: Found {len(all_expected)} {expected_extension} files across the dataset")
            return True
        print(f"[WARNING] No {expected_extension} files found in dataset")
        return False

    all_files = [path for path in local_path.rglob("*") if path.is_file()]
    if all_files:
        print(f"[INFO] Verified: Found {len(all_files)} files across the dataset")
        return True

    print("[WARNING] No files found in dataset")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download a dataset from Hugging Face Hub using a preset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hmdb51",
        choices=sorted(DATASET_PRESETS.keys()),
        help="Dataset preset to download (default: hmdb51).",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Override the Hugging Face repo ID for a custom dataset.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Local directory to save the dataset (default: preset-specific directory).",
    )
    parser.add_argument(
        "--expected_extension",
        type=str,
        default=None,
        help="Override the expected file extension used during verification, for example .avi.",
    )
    args = parser.parse_args()

    preset = DATASET_PRESETS[args.dataset]
    repo_id = args.repo_id or preset["repo_id"]
    repo_type = preset.get("repo_type", "dataset")
    local_dir = args.local_dir or preset.get("default_local_dir", "./data/downloaded_dataset")
    expected_extension = normalize_extension(args.expected_extension or preset.get("expected_extension"))

    print("[START] Dataset Download Script")
    print("=" * 60)
    print(f"[INFO] Preset: {args.dataset}")

    # Download
    download_dataset(repo_id, local_dir, repo_type=repo_type)

    print("\n" + "=" * 60)
    print("[VERIFY] Checking dataset structure...")
    print("=" * 60)

    # Verify
    is_valid = verify_structure(local_dir, expected_extension=expected_extension)

    print("\n" + "=" * 60)
    if is_valid:
        print("[SUCCESS] Dataset ready for training!")
        print(f"[INFO] Use with: --train_dir {local_dir}")
        print("[INFO] Compatible with 3D NEPA pretraining via run_nepa_3d.py")
    else:
        print("[FAIL] Dataset verification failed. Please check the output above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()