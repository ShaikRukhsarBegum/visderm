"""
Evaluate the seven privacy mechanisms (Table 2) at ε=8, δ=1e-5.

Mechanisms:
    Uniform DP   — full-token noise on all 196 patch tokens
    LSP-DP       — attention-weighted noise allocation
    PB-96/64/48  — Privacy Bottleneck at three bottleneck dimensions
    STP-49/98    — Stochastic Token Pruning at two retention sizes

Usage:
    python scripts/eval_privacy.py --checkpoint model1.pth
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
import numpy as np

from src.model import SplitViT
from src.privacy import STPDP, PrivacyBottleneck, UniformDP, LSPDP
from src.data import (
    HAM10000Dataset, load_metadata, patient_grouped_split,
    standard_transforms, MEL_INDEX,
)


def evaluate_mechanism(model, mechanism, test_loader, device, n_draws=10,
                       requires_attention=False):
    """
    Evaluate a privacy mechanism with optional draw averaging.

    For stochastic mechanisms (STP-DP), averaging across multiple draws
    reduces variance in the per-sample prediction without compromising
    privacy (each draw is itself a valid (ε, δ)-DP sample).
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            features, attention = model.client_forward(imgs)
            logit_sum = torch.zeros(imgs.size(0), 7, device=device)
            for _ in range(n_draws):
                if requires_attention:
                    privatized = mechanism(features, attention)
                else:
                    privatized = mechanism(features)
                logit_sum += model.server_forward(privatized)
            logits = logit_sum / n_draws
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    accuracy = 100.0 * np.mean(preds == labels)
    mel_mask = labels == MEL_INDEX
    mel_recall = 100.0 * np.sum((preds == MEL_INDEX) & mel_mask) / max(mel_mask.sum(), 1)
    return accuracy, mel_recall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/HAM10000")
    p.add_argument("--checkpoint", default="model1.pth")
    p.add_argument("--epsilon", type=float, default=8.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--n-draws", type=int, default=10)
    p.add_argument("--output", default="results/privacy_mechanisms.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Setup
    metadata = load_metadata(str(Path(args.data_root) / "HAM10000_metadata.csv"))
    _, _, test_df = patient_grouped_split(metadata, random_state=42)
    test_loader = DataLoader(
        HAM10000Dataset(test_df, str(Path(args.data_root) / "images"),
                        standard_transforms(train=False)),
        batch_size=32, shuffle=False, num_workers=2, pin_memory=True,
    )
    model = SplitViT(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    mechanisms = {
        "Uniform_DP": (UniformDP(args.epsilon, args.delta).to(device), False),
        "LSP_DP":     (LSPDP(args.epsilon, args.delta).to(device), True),
        "PB_96":      (PrivacyBottleneck(192, 96, args.epsilon, args.delta).to(device), False),
        "PB_64":      (PrivacyBottleneck(192, 64, args.epsilon, args.delta).to(device), False),
        "PB_48":      (PrivacyBottleneck(192, 48, args.epsilon, args.delta).to(device), False),
        "STP_98":     (STPDP(98, args.epsilon, args.delta).to(device), False),
        "STP_49":     (STPDP(49, args.epsilon, args.delta).to(device), False),
    }

    results = {}
    print(f"\n{'Mechanism':<12} {'Acc (%)':>10} {'Mel (%)':>10} {'Payload':>10}")
    print("-" * 50)
    for name, (mech, needs_attn) in mechanisms.items():
        acc, mel = evaluate_mechanism(model, mech, test_loader, device,
                                      n_draws=args.n_draws,
                                      requires_attention=needs_attn)
        # Payload calculation
        if name.startswith("STP"):
            n_keep = int(name.split("_")[1])
            payload_kb = (n_keep + 1) * 192 * 4 / 1024
        elif name.startswith("PB"):
            d = int(name.split("_")[1])
            payload_kb = 197 * d * 4 / 1024
        else:
            payload_kb = 197 * 192 * 4 / 1024  # full
        results[name] = {"acc": acc, "mel": mel, "payload_kb": round(payload_kb, 1)}
        print(f"{name:<12} {acc:>9.2f}% {mel:>9.2f}% {payload_kb:>8.0f} KB")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
