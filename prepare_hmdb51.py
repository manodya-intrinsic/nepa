import argparse
import glob
import os
import shutil
import subprocess
import urllib.request

from datasets import ClassLabel, Dataset, DatasetDict, Video


HMDB51_VIDEOS_URL = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
HMDB51_SPLITS_URL = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cmd(command: list[str]) -> None:
    print("[CMD]", " ".join(command))
    subprocess.run(command, check=True)


def download_file(url: str, target_path: str, force: bool = False) -> None:
    if os.path.exists(target_path) and not force:
        print(f"[SKIP] Exists: {target_path}")
        return
    ensure_dir(os.path.dirname(target_path))
    print(f"[DOWNLOAD] {url} -> {target_path}")
    urllib.request.urlretrieve(url, target_path)


def require_unrar() -> None:
    if shutil.which("unrar") is None:
        raise RuntimeError(
            "'unrar' command not found. Install it first. On Colab, run: !apt-get -y install unrar"
        )


def extract_rar(rar_path: str, output_dir: str) -> None:
    ensure_dir(output_dir)
    run_cmd(["unrar", "x", "-o+", rar_path, output_dir])


def extract_class_archives(class_rars_dir: str, videos_root: str) -> None:
    class_rars = sorted(glob.glob(os.path.join(class_rars_dir, "*.rar")))
    if not class_rars:
        raise FileNotFoundError(f"No class .rar files found under: {class_rars_dir}")

    print(f"[INFO] Found {len(class_rars)} class archives")
    for idx, rar_path in enumerate(class_rars, start=1):
        cls_name = os.path.splitext(os.path.basename(rar_path))[0]
        cls_out = os.path.join(videos_root, cls_name)
        ensure_dir(cls_out)
        existing_videos = glob.glob(os.path.join(cls_out, "*.avi"))
        if existing_videos:
            print(f"[SKIP {idx}/{len(class_rars)}] {cls_name} already extracted")
            continue
        print(f"[EXTRACT {idx}/{len(class_rars)}] {cls_name}")
        extract_rar(rar_path, cls_out)


def parse_split_file(split_path: str) -> list[tuple[str, int]]:
    rows = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            file_name, tag = line.split()
            rows.append((file_name, int(tag)))
    return rows


def build_dataset_dict(videos_root: str, splits_root: str, fold: int, val_ratio: float, seed: int) -> DatasetDict:
    split_files = sorted(glob.glob(os.path.join(splits_root, f"*_test_split{fold}.txt")))
    if not split_files:
        raise FileNotFoundError(f"No split files found for fold {fold} in: {splits_root}")

    classes = sorted(
        os.path.basename(path).replace(f"_test_split{fold}.txt", "") for path in split_files
    )
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    for split_path in split_files:
        cls_name = os.path.basename(split_path).replace(f"_test_split{fold}.txt", "")
        label_id = class_to_id[cls_name]
        for file_name, tag in parse_split_file(split_path):
            video_path = os.path.join(videos_root, cls_name, file_name)
            if not os.path.exists(video_path):
                continue
            if tag == 1:
                train_paths.append(video_path)
                train_labels.append(label_id)
            elif tag == 2:
                test_paths.append(video_path)
                test_labels.append(label_id)

    label_feature = ClassLabel(names=classes)

    train_ds = Dataset.from_dict({"video": train_paths, "label": train_labels})
    train_ds = train_ds.cast_column("label", label_feature)
    train_ds = train_ds.cast_column("video", Video(decode=False))

    test_ds = Dataset.from_dict({"video": test_paths, "label": test_labels})
    test_ds = test_ds.cast_column("label", label_feature)
    test_ds = test_ds.cast_column("video", Video(decode=False))

    train_val = train_ds.train_test_split(
        test_size=val_ratio,
        seed=seed,
        stratify_by_column="label",
    )

    return DatasetDict(
        {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": test_ds,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HMDB51 as a Hugging Face DatasetDict.")
    parser.add_argument("--work_dir", default="data/hmdb51_work", help="Temporary working directory.")
    parser.add_argument(
        "--output_dir",
        default="data/hmdb51-hf",
        help="Output directory for save_to_disk DatasetDict.",
    )
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3], help="HMDB51 split fold.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio from train split.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for split.")
    parser.add_argument("--force_download", action="store_true", help="Re-download archive files.")
    args = parser.parse_args()

    require_unrar()

    raw_dir = os.path.join(args.work_dir, "raw")
    extract_dir = os.path.join(args.work_dir, "extract")
    videos_dir = os.path.join(args.work_dir, "videos")
    splits_dir = os.path.join(args.work_dir, "splits")

    ensure_dir(raw_dir)
    ensure_dir(extract_dir)
    ensure_dir(videos_dir)
    ensure_dir(splits_dir)

    videos_rar = os.path.join(raw_dir, "hmdb51_org.rar")
    splits_rar = os.path.join(raw_dir, "test_train_splits.rar")

    download_file(HMDB51_VIDEOS_URL, videos_rar, force=args.force_download)
    download_file(HMDB51_SPLITS_URL, splits_rar, force=args.force_download)

    # Extract top-level archives.
    extract_rar(videos_rar, extract_dir)
    extract_rar(splits_rar, splits_dir)

    # Extract per-class video archives.
    extract_class_archives(extract_dir, videos_dir)

    # Build and save HF dataset dict.
    ds = build_dataset_dict(
        videos_root=videos_dir,
        splits_root=splits_dir,
        fold=args.fold,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    ensure_dir(os.path.dirname(args.output_dir) or ".")
    print(f"[SAVE] Writing DatasetDict to: {args.output_dir}")
    ds.save_to_disk(args.output_dir)
    print(ds)
    print("[DONE] HMDB51 dataset prepared successfully")


if __name__ == "__main__":
    main()
