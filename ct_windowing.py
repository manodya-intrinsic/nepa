#!/usr/bin/env python3
"""
Windowing utilities for CT data preprocessing.

CT scans store intensity as Hounsfield Units (HU):
  - Air: -1000 HU
  - Fat: -100 HU
  - Soft tissue: 0-100 HU
  - Bone: 300-1000+ HU
  - Metal: >1000 HU

Windowing extracts clinically relevant intensity ranges.
"""

import numpy as np


def apply_ct_window(vol: np.ndarray, center: int = 40, width: int = 400) -> np.ndarray:
    """
    Apply CT windowing to Hounsfield Unit data.

    Args:
        vol: CT volume in Hounsfield Units (float or int)
        center: Window center (level) in HU
        width: Window width in HU

    Returns:
        Windowed volume normalized to [0, 255] as uint8

    Example:
        # Soft tissue window (default)
        windowed = apply_ct_window(vol)  # center=40, width=400

        # Liver window
        windowed = apply_ct_window(vol, center=60, width=150)

        # Bone window
        windowed = apply_ct_window(vol, center=400, width=1500)
    """
    lower = center - width / 2
    upper = center + width / 2

    # Clip to window range
    windowed = np.clip(vol, lower, upper)

    # Normalize to [0, 1] then [0, 255]
    windowed = (windowed - lower) / width * 255

    return windowed.astype(np.uint8)


def apply_multi_window(vol: np.ndarray, windows: list = None) -> np.ndarray:
    """
    Apply multiple windows and stack as multi-channel output.

    Args:
        vol: CT volume in HU
        windows: List of (center, width) tuples

    Returns:
        Multi-window volume of shape (H, W, D, C)
        where C = number of windows

    Example:
        windows = [
            (40, 400),   # Soft tissue
            (60, 150),   # Liver
            (400, 1500), # Bone
        ]
        multi = apply_multi_window(vol, windows)
        # Output: (512, 512, 200, 3)
    """
    if windows is None:
        windows = [
            (40, 400),   # Soft tissue
            (60, 150),   # Liver
        ]

    windowed_list = []
    for center, width in windows:
        w = apply_ct_window(vol, center=center, width=width)
        windowed_list.append(w)

    return np.stack(windowed_list, axis=-1)


def compare_normalization_methods(vol_sample: np.ndarray):
    """
    Compare different normalization approaches.

    Args:
        vol_sample: Sample CT slice or volume

    Returns:
        Dictionary with different normalized versions
    """
    results = {}

    # Method 1: Min-Max (current default)
    min_max = vol_sample - vol_sample.min()
    if min_max.max() > 0:
        min_max = min_max / min_max.max()
    min_max = (min_max * 255).astype(np.uint8)
    results['min_max'] = min_max

    # Method 2: Soft tissue window
    soft_tissue = apply_ct_window(vol_sample, center=40, width=400)
    results['soft_tissue'] = soft_tissue

    # Method 3: Liver window
    liver = apply_ct_window(vol_sample, center=60, width=150)
    results['liver'] = liver

    # Method 4: Multi-window
    multi = apply_multi_window(vol_sample, windows=[(40, 400), (60, 150)])
    results['multi_window'] = multi

    return results


# Statistics from different methods
def print_statistics(vol_sample: np.ndarray):
    """Print statistics for different normalization methods."""
    results = compare_normalization_methods(vol_sample)

    print("Normalization Method Comparison")
    print("=" * 60)

    for method_name, normalized in results.items():
        print(f"\n{method_name}:")
        print(f"  Shape: {normalized.shape}")
        print(f"  Min: {normalized.min()}, Max: {normalized.max()}")
        print(f"  Mean: {normalized.mean():.2f}, Std: {normalized.std():.2f}")
        print(f"  Dtype: {normalized.dtype}")


if __name__ == "__main__":
    # Example: Create sample CT-like volume
    vol = np.random.randint(-100, 200, (512, 512, 10), dtype=np.float32)

    # Test windowing
    windowed = apply_ct_window(vol, center=40, width=400)
    print(f"Original HU range: [{vol.min():.0f}, {vol.max():.0f}]")
    print(f"Windowed range: [{windowed.min()}, {windowed.max()}]")
    print(f"Output shape: {windowed.shape}")
