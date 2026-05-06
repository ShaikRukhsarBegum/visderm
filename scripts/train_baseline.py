"""
Train Split-ViT from ImageNet pretraining and reproduce Table 1.

Usage:
    python scripts/train_baseline.py --seed 42
    python scripts/train_baseline.py --seeds 42 123 777 2024 31415  # multi-seed (Table 3)

Requires HAM10000 in data/HAM10000/ (see README).
"""
import argparse
import json
from pathlib import Path

# Ensure project root is on PYTHONPATH so `import src.*` works
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import train_split_vit, evaluate
from src.data import (
    HAM10000Dataset, load_metadata, patient_grouped_split,
    standard_transforms,
)
from src.model import SplitViT
import torch
from torch.utils.data import DataLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/HAM10000")
    p.add_argument("--output-dir", default="runs/table1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Multiple seeds for Table 3 multi-seed evaluation")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    data_root = Path(args.data_root)
    metadata_csv = str(data_root / "HAM10000_metadata.csv")
    images_dir = str(data_root / "images")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds if args.seeds is not None else [args.seed]
    results = {}

    for seed in seeds:
        print(f"\n{'='*70}\nTraining seed={seed}\n{'='*70}")
        ckpt_path = train_split_vit(
            metadata_csv=metadata_csv,
            images_dir=images_dir,
            output_dir=str(output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
            device=args.device,
        )

        # Final test evaluation
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        metadata = load_metadata(metadata_csv)
        _, _, test_df = patient_grouped_split(metadata, random_state=seed)
        test_loader = DataLoader(
            HAM10000Dataset(test_df, images_dir, standard_transforms(train=False)),
            batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
        )
        model = SplitViT().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        acc, mel, per_class = evaluate(model, test_loader, device)
        results[seed] = {
            "test_acc": acc, "test_mel": mel,
            "per_class_recall": per_class.tolist(),
            "checkpoint": str(ckpt_path),
        }

    # Save aggregated results
    out_json = output_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Multi-seed summary
    if len(seeds) > 1:
        accs = [r["test_acc"] for r in results.values()]
        mels = [r["test_mel"] for r in results.values()]
        import statistics
        print(f"\n{'='*70}\nMulti-seed summary (n={len(seeds)} seeds)\n{'='*70}")
        print(f"  Accuracy:      {statistics.mean(accs):.2f}% ± {statistics.stdev(accs):.2f}%")
        print(f"  Melanoma rec:  {statistics.mean(mels):.2f}% ± {statistics.stdev(mels):.2f}%")
        print(f"  Paper Table 3:  73.6% ± 1.1% accuracy, 79.6% ± 3.7% mel recall")

    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
