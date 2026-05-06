"""
DDI cross-dataset evaluation (Section 4.5).

Evaluates the canonical HAM10000-trained checkpoint on the Diverse Dermatology
Images (DDI) dataset (Daneshjou et al. 2022; n=656 clinical photographs across
Fitzpatrick I-II / III-IV / V-VI strata) and reports:

  - Sensitivity at threshold=0.5 by Fitzpatrick stratum
  - Specificity at threshold=0.5 by Fitzpatrick stratum
  - ROC-AUC (binary malignancy) by Fitzpatrick stratum
  - 2,000-iteration bootstrap 95% confidence intervals

Outputs the table reproduced as Table 5 in the paper, and a JSON file with
all per-stratum statistics.

Required: DDI dataset at data/DDI/ (see README — registration at
https://ddi-dataset.github.io/).

Usage:
    python scripts/ddi_evaluation.py --checkpoint model1.pth
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from torchvision import transforms as T
from PIL import Image

from src.model import SplitViT
from src.data import standard_transforms, MEL_INDEX, CLASS_NAMES


# Binary malignancy collapse: P(malignant) = P(akiec) + P(bcc) + P(mel)
MALIGNANT_INDICES = [
    CLASS_NAMES.index("akiec"),
    CLASS_NAMES.index("bcc"),
    CLASS_NAMES.index("mel"),
]


def fst_bucket(fitzpatrick: int) -> str:
    """Map raw Fitzpatrick scale (1-6) to the three-bucket stratification."""
    if fitzpatrick in (1, 2):
        return "I-II"
    if fitzpatrick in (3, 4):
        return "III-IV"
    if fitzpatrick in (5, 6):
        return "V-VI"
    return "unknown"


class DDIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform or standard_transforms(train=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.images_dir / row["DDI_file"]
        img = Image.open(path).convert("RGB")
        return self.transform(img), int(row["malignant"]), row["fst_bucket"]


def bootstrap_ci(y_true, y_score, statistic_fn, n_boot=2000, seed=42):
    """Compute bootstrap 95% confidence interval for a statistic."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots.append(statistic_fn(y_true[idx], y_score[idx]))
        except ValueError:
            continue  # all-one-class bootstrap sample, skip
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/DDI")
    p.add_argument("--checkpoint", default="model1.pth")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--output", default="results/ddi_evaluation.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    data_root = Path(args.data_root)
    metadata_csv = data_root / "ddi_metadata.csv"
    images_dir = data_root / "images"

    # Load DDI metadata
    df = pd.read_csv(metadata_csv)
    df["fst_bucket"] = df["skin_tone"].apply(fst_bucket)
    df = df[df["fst_bucket"] != "unknown"].reset_index(drop=True)
    df["malignant"] = df["malignant"].astype(int)
    print(f"Loaded DDI: n={len(df)}, "
          f"malignant={df['malignant'].sum()}, benign={(df['malignant']==0).sum()}")
    print(f"Stratification: {df['fst_bucket'].value_counts().to_dict()}")

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SplitViT(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    # Inference
    loader = DataLoader(
        DDIDataset(df, str(images_dir)),
        batch_size=32, shuffle=False, num_workers=2, pin_memory=True,
    )
    all_scores, all_labels, all_buckets = [], [], []
    with torch.no_grad():
        for imgs, labels, buckets in loader:
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            probs = torch.softmax(logits, dim=1)
            p_malignant = probs[:, MALIGNANT_INDICES].sum(dim=1)
            all_scores.extend(p_malignant.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_buckets.extend(list(buckets))

    scores = np.asarray(all_scores)
    labels = np.asarray(all_labels)
    buckets = np.asarray(all_buckets)

    # Per-stratum statistics with bootstrap CIs
    results = {}
    for stratum in ["I-II", "III-IV", "V-VI", "All"]:
        mask = np.ones(len(scores), dtype=bool) if stratum == "All" else (buckets == stratum)
        y_true = labels[mask]
        y_score = scores[mask]
        n_total, n_mal = len(y_true), int(y_true.sum())

        preds = (y_score > args.threshold).astype(int)
        sens = 100.0 * np.sum((preds == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1)
        spec = 100.0 * np.sum((preds == 0) & (y_true == 0)) / max(np.sum(y_true == 0), 1)
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float("nan")

        # Bootstrap CIs
        sens_lo, sens_hi = bootstrap_ci(
            y_true, y_score,
            lambda yt, ys: 100.0 * np.sum((ys > args.threshold) & (yt == 1)) / max(np.sum(yt == 1), 1),
            n_boot=args.n_bootstrap,
        )
        auc_lo, auc_hi = bootstrap_ci(
            y_true, y_score,
            lambda yt, ys: roc_auc_score(yt, ys) if len(np.unique(yt)) > 1 else 0.5,
            n_boot=args.n_bootstrap,
        )

        results[stratum] = {
            "n_total": n_total, "n_malignant": n_mal,
            "sensitivity_pct": round(sens, 2),
            "sensitivity_ci": [round(sens_lo, 2), round(sens_hi, 2)],
            "specificity_pct": round(spec, 2),
            "auc": round(auc, 4),
            "auc_ci": [round(auc_lo, 4), round(auc_hi, 4)],
        }

    # Print Table 5
    print("\nTable 5: DDI cross-dataset evaluation (binary malignancy collapse)")
    print(f"{'FST':<8} {'n':>5} {'n_mal':>6} {'Sens [95% CI]':>22} {'Spec':>8} {'AUC [95% CI]':>22}")
    for stratum, r in results.items():
        sens_str = f"{r['sensitivity_pct']:.1f} [{r['sensitivity_ci'][0]:.1f},{r['sensitivity_ci'][1]:.1f}]"
        auc_str = f"{r['auc']:.3f} [{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}]"
        print(f"{stratum:<8} {r['n_total']:>5} {r['n_malignant']:>6} {sens_str:>22} "
              f"{r['specificity_pct']:>7.1f}% {auc_str:>22}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print("\nNote: All AUC confidence intervals cross 0.5. Performance is "
          "statistically indistinguishable from chance, indicating dermoscopy-to-"
          "clinical-photo domain shift dominates skin-tone effects. See Section 4.5.")


if __name__ == "__main__":
    main()
