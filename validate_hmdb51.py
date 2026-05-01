#!/usr/bin/env python3
"""
Validate HMDB51 dataset and report corruption stats.
Identifies videos with 0 frames, decode failures, bad shapes, etc.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset, load_from_disk
from torchvision.io import read_video

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def validate_video(video_path: str) -> dict:
    """
    Try to decode a video and return its status.
    
    Returns:
        dict with keys:
            - valid: bool
            - frames: int or None
            - error: str or None
    """
    try:
        video, _, _ = read_video(video_path, pts_unit="sec")
    except Exception as e:
        return {"valid": False, "frames": None, "error": str(e)[:100]}
    
    if video.ndim != 4:
        return {
            "valid": False,
            "frames": None,
            "error": f"Bad shape {tuple(video.shape)}, expected [T,H,W,C]",
        }
    
    if video.shape[0] == 0:
        return {"valid": False, "frames": 0, "error": "0 frames"}
    
    return {"valid": True, "frames": video.shape[0], "error": None}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate HMDB51 dataset corruption.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Path or hub ID of dataset to validate.",
    )
    parser.add_argument(
        "--load_from_disk",
        type=bool,
        default=False,
        help="Load from local disk (vs. hub).",
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="Name of video column.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to validate (train or validation).",
    )
    parser.add_argument(
        "--max_samples",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=None,
        help="Max samples to check (None = all).",
    )
    
    args = parser.parse_args()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_name}...")
    if args.load_from_disk:
        dataset_dict = load_from_disk(args.dataset_name)
    else:
        dataset_dict = load_dataset(args.dataset_name)
    
    if args.split not in dataset_dict:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(dataset_dict.keys())}")
    
    dataset = dataset_dict[args.split]
    logger.info(f"Dataset has {len(dataset)} examples.")
    
    # Optionally limit samples
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    # Validate
    logger.info(f"Validating {len(dataset)} videos...")
    valid_count = 0
    corrupt_count = 0
    frame_counts = []
    error_types = {}
    
    corrupt_examples = []
    
    for idx, example in enumerate(dataset):
        if idx % 500 == 0:
            logger.info(f"  [{idx}/{len(dataset)}]...")
        
        video_entry = example[args.video_column]
        
        # Resolve video path
        if isinstance(video_entry, dict):
            video_path = video_entry.get("path")
        elif hasattr(video_entry, "path"):
            video_path = video_entry.path
        else:
            video_path = str(video_entry)
        
        result = validate_video(video_path)
        
        if result["valid"]:
            valid_count += 1
            frame_counts.append(result["frames"])
        else:
            corrupt_count += 1
            error = result["error"]
            error_types[error] = error_types.get(error, 0) + 1
            
            # Keep first 10 corrupt examples for user reference
            if len(corrupt_examples) < 10:
                corrupt_examples.append({
                    "index": idx,
                    "path": video_path,
                    "error": error,
                })
    
    # Report stats
    logger.info("\n" + "="*80)
    logger.info("VALIDATION REPORT")
    logger.info("="*80)
    logger.info(f"Total videos: {len(dataset)}")
    logger.info(f"Valid: {valid_count} ({100*valid_count/len(dataset):.1f}%)")
    logger.info(f"Corrupt: {corrupt_count} ({100*corrupt_count/len(dataset):.1f}%)")
    
    if frame_counts:
        logger.info(f"\nFrame count stats (valid videos only):")
        logger.info(f"  Min: {min(frame_counts)}")
        logger.info(f"  Max: {max(frame_counts)}")
        logger.info(f"  Mean: {sum(frame_counts)/len(frame_counts):.1f}")
        logger.info(f"  Median: {sorted(frame_counts)[len(frame_counts)//2]}")
    
    if error_types:
        logger.info(f"\nCorruption types:")
        for error, count in sorted(error_types.items(), key=lambda x: -x[1]):
            logger.info(f"  {error}: {count} ({100*count/corrupt_count:.1f}%)")
    
    if corrupt_examples:
        logger.info(f"\nFirst {len(corrupt_examples)} corrupt examples:")
        for ex in corrupt_examples:
            logger.info(f"  [{ex['index']}] {ex['path']}")
            logger.info(f"         Error: {ex['error']}")
    
    logger.info("="*80)
    
    return corrupt_count == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
