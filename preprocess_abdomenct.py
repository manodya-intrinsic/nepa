"""
Preprocess AbdomenCT-1K NIfTI volumes into 2D slice tensors for NEPA pretraining.

Pipeline (matches the diagram):
  1. Load NIfTI volume (.nii.gz) in HU
  2. Canonical orientation (RAS+)        -- prevents flipped volumes
  3. Resample to isotropic in-plane spacing (1.0 x 1.0 mm)
  4. Apply soft-tissue window (-160, +240 HU) -> [0, 1]
  5. Slice along depth axis (axial)
  6. Filter empty slices (foreground < 5%)
  7. Resize to 224 x 224
  8. Save kept slices as a single float16 .npy per case
"""

import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


# --------- preprocessing parameters ---------
TARGET_SPACING_MM = 1.0      # isotropic in-plane spacing in mm
WINDOW_LOWER_HU   = -160     # soft-tissue window lower bound
WINDOW_UPPER_HU   = 240      # soft-tissue window upper bound
OUTPUT_SIZE       = 224      # final spatial size for ViT
FG_THRESHOLD      = 0.05     # drop slices with foreground < 5%
# --------------------------------------------


def load_volume(nii_path):
    """Load NIfTI -> (D, H, W) float32 array in HU + in-plane voxel spacing."""
    img = nib.load(nii_path)
    img = nib.as_closest_canonical(img)         # standardize orientation
    data = img.get_fdata().astype(np.float32)   # (X, Y, Z) in nibabel default
    # Transpose to (D, H, W) so the first axis is the slicing axis (axial)
    data = np.transpose(data, (2, 1, 0))
    spacing = img.header.get_zooms()            # (sx, sy, sz) in mm
    # After transpose, in-plane spacing comes from the original Y and X axes
    in_plane_spacing = (float(spacing[1]), float(spacing[0]))
    return data, in_plane_spacing


def resample_in_plane(volume, spacing, target=TARGET_SPACING_MM):
    """Resample H, W axes to target spacing using bilinear interpolation."""
    zoom_h = spacing[0] / target
    zoom_w = spacing[1] / target
    if abs(zoom_h - 1.0) < 1e-3 and abs(zoom_w - 1.0) < 1e-3:
        return volume
    # order=1 = bilinear, fast and good enough for CT
    return zoom(volume, (1.0, zoom_h, zoom_w), order=1, prefilter=False)


def apply_window(volume, lower=WINDOW_LOWER_HU, upper=WINDOW_UPPER_HU):
    """Clip HU to soft-tissue window and rescale to [0, 1]."""
    volume = np.clip(volume, lower, upper)
    volume = (volume - lower) / (upper - lower)
    return volume.astype(np.float32)


def is_informative_slice(slice_2d, threshold=FG_THRESHOLD):
    """A slice is kept if more than `threshold` fraction is non-air."""
    fg_fraction = float((slice_2d > 0.05).mean())
    return fg_fraction > threshold


def resize_slice(slice_2d, size=OUTPUT_SIZE):
    """Resize a 2D slice to (size, size) using bilinear interpolation."""
    h, w = slice_2d.shape
    zoom_h = size / h
    zoom_w = size / w
    return zoom(slice_2d, (zoom_h, zoom_w), order=1, prefilter=False).astype(np.float32)


def preprocess_volume(nii_path):
    """Full pipeline for one volume. Returns (N_kept, 224, 224) float16 array."""
    volume, spacing = load_volume(nii_path)
    volume = resample_in_plane(volume, spacing)
    volume = apply_window(volume)

    kept_slices = []
    for i in range(volume.shape[0]):
        s = volume[i]
        if not is_informative_slice(s):
            continue
        s = resize_slice(s)
        kept_slices.append(s)

    if not kept_slices:
        return None

    out = np.stack(kept_slices, axis=0).astype(np.float16)  # float16 saves disk
    return out


def preprocess_dataset(input_dir, output_dir):
    """Process every .nii.gz in input_dir and save .npy per case."""
    os.makedirs(output_dir, exist_ok=True)

    nii_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.nii.gz"), recursive=True)
        + glob.glob(os.path.join(input_dir, "**", "*.nii"), recursive=True)
    )

    print(f"Found {len(nii_files)} NIfTI files")
    total_slices = 0
    failed = 0

    for idx, path in enumerate(nii_files):
        case_id = os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")
        out_path = os.path.join(output_dir, f"{case_id}.npy")

        if os.path.exists(out_path):
            print(f"[{idx + 1}/{len(nii_files)}] {case_id}: already done")
            continue

        try:
            slices = preprocess_volume(path)
            if slices is None:
                print(f"[{idx + 1}/{len(nii_files)}] {case_id}: no informative slices")
                failed += 1
                continue
            np.save(out_path, slices)
            total_slices += slices.shape[0]
            print(f"[{idx + 1}/{len(nii_files)}] {case_id}: kept {slices.shape[0]} slices")
        except Exception as e:
            print(f"[{idx + 1}/{len(nii_files)}] {case_id}: FAILED - {e}")
            failed += 1

    print(f"\nDone. Total slices kept: {total_slices}, failed cases: {failed}")


if __name__ == "__main__":
    if __name__ == "__main__":
    INPUT_DIR  = "/content/drive/MyDrive/nepa_medical/data/data/ImagePart1"
    OUTPUT_DIR = "/content/preprocessed_slices"   # local — fast
    preprocess_dataset(INPUT_DIR, OUTPUT_DIR)

    # Persist to Drive as a single zip
    import shutil
    print("Zipping to Drive...")
    shutil.make_archive(
        "/content/drive/MyDrive/nepa_medical/preprocessed_slices",
        "zip",
        OUTPUT_DIR
    )
    print("Done. Zip saved to Drive.")