"""
Single-file reproduction of Table 1 (Split-ViT k=6) from the VisDerm paper.

This script:
  1. Loads HAM10000 metadata from ``data/HAM10000/HAM10000_metadata.csv``
  2. Applies the canonical patient-grouped 70/15/15 split
     (GroupShuffleSplit on lesion_id, random_state=42)
  3. Loads the released checkpoint ``model1.pth`` (or downloads it from GitHub Releases
     if not present locally)
  4. Evaluates on the test split (n=1527) and prints accuracy, melanoma recall

Expected output:
    Test accuracy:    73.87%
    Melanoma recall:  79.57%

Usage:
    python reproduce.py [--data-root data/HAM10000] [--checkpoint model1.pth]
"""
import argparse
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import (
    HAM10000Dataset,
    load_metadata,
    patient_grouped_split,
    standard_transforms,
)
from src.model import SplitViT
from src.train import evaluate


# Canonical checkpoint hash (from the original training run)
CHECKPOINT_SHA256 = "c32d8680d8a56524e1e99f2929cc2c56f05a8aa0169ed00484059ff511a6e09e"

# GitHub Release URL for the canonical checkpoint
CHECKPOINT_URL = "https://github.com/ShaikRukhsarBegum/visderm/releases/download/v1.0-ijai/model1.pth"


def file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Table 1 (Split-ViT k=6) from the VisDerm paper."
    )
    parser.add_argument(
        "--data-root", default="data/HAM10000",
        help="Path to HAM10000 dataset directory (default: data/HAM10000)",
    )
    parser.add_argument(
        "--checkpoint", default="model1.pth",
        help="Path to released model1.pth (default: model1.pth)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Evaluation batch size (default: 32)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--skip-hash-check", action="store_true",
        help="Skip SHA-256 verification of the checkpoint",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    metadata_csv = data_root / "HAM10000_metadata.csv"
    images_dir = data_root / "images"
    checkpoint_path = Path(args.checkpoint)

    # ------------------------------------------------------------------
    # 1. Data preparation
    # ------------------------------------------------------------------
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"HAM10000 metadata not found at {metadata_csv}.\n"
            f"Download from https://doi.org/10.7910/DVN/DBW86T and place at "
            f"{data_root}/HAM10000_metadata.csv with images in {images_dir}/"
        )
    if not images_dir.exists():
        raise FileNotFoundError(
            f"HAM10000 images directory not found at {images_dir}.\n"
            f"Ensure the .jpg files are placed in {images_dir}/"
        )

    print(f"Loading HAM10000 metadata: {metadata_csv}")
    metadata = load_metadata(str(metadata_csv))
    print(f"  Total images: {len(metadata)}")

    train_df, val_df, test_df = patient_grouped_split(metadata, random_state=42)
    print(f"  Patient-grouped split: train={len(train_df)} "
          f"val={len(val_df)} test={len(test_df)}")
    assert len(test_df) == 1527, (
        f"Expected canonical test split of 1527 images; got {len(test_df)}. "
        f"Please verify random_state=42 reproducibility."
    )

    # ------------------------------------------------------------------
    # 2. Checkpoint
    # ------------------------------------------------------------------
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}.\n"
            f"Download from {CHECKPOINT_URL} and place at the project root."
        )

    if not args.skip_hash_check:
        actual_hash = file_sha256(checkpoint_path)
        if actual_hash != CHECKPOINT_SHA256:
            print(
                f"WARNING: Checkpoint SHA-256 mismatch.\n"
                f"  Expected: {CHECKPOINT_SHA256}\n"
                f"  Got:      {actual_hash}\n"
                f"This may produce different numbers from the paper. "
                f"Use --skip-hash-check to suppress this warning."
            )
        else:
            print(f"Checkpoint SHA-256 verified: {CHECKPOINT_SHA256[:16]}...")

    # ------------------------------------------------------------------
    # 3. Model + evaluation
    # ------------------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SplitViT(num_classes=7, split_point=6, pretrained=False).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    test_loader = DataLoader(
        HAM10000Dataset(test_df, str(images_dir), standard_transforms(train=False)),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    print(f"\nEvaluating on n={len(test_df)} test images ...")
    test_acc, test_mel, per_class = evaluate(model, test_loader, device)

    # ------------------------------------------------------------------
    # 4. Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Table 1 (Split-ViT k=6) — Reproduction Results")
    print("=" * 60)
    print(f"  Test accuracy:    {test_acc:.2f}%   (paper: 73.87%)")
    print(f"  Melanoma recall:  {test_mel:.2f}%   (paper: 79.57%)")
    print("\nPer-class recall (%):")
    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    for cls, recall in zip(class_names, per_class.tolist()):
        marker = " <-- melanoma" if cls == "mel" else ""
        print(f"  {cls:<8} {recall:5.2f}%{marker}")

    print("\nPer-inference payload:")
    print("  Without STP-DP:  148 KB  (full intermediate representation)")
    print("  With STP-DP:      38 KB  (49 of 196 patch tokens, ε=8, δ=1e-5)")


if __name__ == "__main__":
    main()
