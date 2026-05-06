"""
Run federated learning experiments with Byzantine attacks (Table 4).

Configuration (Section 4.6 of the paper):
  - 10 clients, 2 malicious, non-IID Dirichlet α=0.5 partition
  - 30 communication rounds, 2 local epochs per client per round
  - Aggregators: FedAvg, AGM v6, History AGM, HistAGM v8
  - Attacks: clean, negate, flip, ALIE

Usage:
    python scripts/run_federated.py
    python scripts/run_federated.py --aggregators FedAvg "History AGM" --attacks clean alie
"""
import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.model import SplitViT
from src.federated import FedAvg, AGMv6, HistoryAGM, HistAGMv8
from src.attacks import negate_attack, alie_attack, flip_labels
from src.data import (
    HAM10000Dataset, load_metadata, patient_grouped_split,
    standard_transforms, class_weights, N_CLASSES, MEL_INDEX,
)
from src.train import evaluate


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float, seed: int):
    """Non-IID partition via per-class Dirichlet sampling."""
    np.random.seed(seed)
    client_indices = [[] for _ in range(n_clients)]
    for c in range(N_CLASSES):
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        proportions = np.random.dirichlet([alpha] * n_clients)
        splits = (proportions * len(idx)).astype(int)
        splits[-1] = len(idx) - splits[:-1].sum()  # absorb rounding error
        start = 0
        for i in range(n_clients):
            client_indices[i].extend(idx[start:start + splits[i]].tolist())
            start += splits[i]
    return client_indices


def fed_train(
    aggregator,
    attack: str,
    train_df, train_ds, val_loader,
    images_dir: str,
    n_clients: int = 10,
    n_byzantine: int = 2,
    n_rounds: int = 30,
    local_epochs: int = 2,
    seed: int = 42,
    device: str = "cuda",
):
    """One full federated training run with the specified aggregator + attack."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    client_indices = dirichlet_partition(
        train_df["label"].values, n_clients, alpha=0.5, seed=seed
    )
    global_model = SplitViT().to(device)
    weights = class_weights(boost_mel_factor=3.0).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    aggregator.reset()

    for round_idx in range(n_rounds):
        updates, honest_updates = [], []
        for c in range(n_clients):
            local_model = SplitViT().to(device)
            local_model.load_state_dict(global_model.state_dict())

            is_byzantine = c >= (n_clients - n_byzantine)

            # Build the client's local dataloader (with optional label flipping)
            if is_byzantine and attack == "flip":
                flipped = train_df.iloc[client_indices[c]].copy()
                flipped["label"] = flip_labels(
                    torch.tensor(flipped["label"].values), N_CLASSES
                ).numpy()
                local_ds = HAM10000Dataset(flipped, images_dir,
                                           standard_transforms(train=True))
            else:
                local_ds = Subset(train_ds, client_indices[c])

            loader = DataLoader(local_ds, batch_size=32, shuffle=True, num_workers=0)

            # Local training
            local_optim = optim.SGD(local_model.parameters(), lr=0.01)
            local_model.train()
            for _ in range(local_epochs):
                for imgs, labels in loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    local_optim.zero_grad()
                    logits, _ = local_model(imgs)
                    loss = criterion(logits, labels)
                    loss.backward()
                    local_optim.step()

            # Compute parameter delta
            update = {
                n: (p2.data - p1.data).cpu()
                for (n, p1), (_, p2) in zip(
                    global_model.named_parameters(), local_model.named_parameters()
                )
            }

            # Apply Byzantine perturbation
            if is_byzantine and attack == "negate":
                update = negate_attack(update, scale=2.0)
            if is_byzantine and attack == "alie" and len(honest_updates) >= 2:
                update = alie_attack(honest_updates, std_factor=1.5)
            if not is_byzantine:
                honest_updates.append(update)
            updates.append(update)

        # Aggregate (flatten -> aggregator -> reshape)
        param_names = list(updates[0].keys())
        flat_updates = [
            torch.cat([u[k].flatten() for k in param_names]) for u in updates
        ]
        agg_flat = aggregator.aggregate(flat_updates)

        # Apply aggregated update to global model
        idx = 0
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                num = p.numel()
                p.data += agg_flat[idx:idx + num].reshape(p.shape).to(device)
                idx += num

        if (round_idx + 1) % 10 == 0:
            val_acc, val_mel, _ = evaluate(global_model, val_loader, device)
            print(f"  R{round_idx+1:02d}: val_acc={val_acc:.2f}% val_mel={val_mel:.2f}%")

    return global_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/HAM10000")
    p.add_argument("--output", default="results/federated.json")
    p.add_argument("--aggregators", nargs="+",
                   default=["FedAvg", "AGM v6", "History AGM", "HistAGM v8"])
    p.add_argument("--attacks", nargs="+",
                   default=["clean", "negate", "flip", "alie"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Data setup
    data_root = Path(args.data_root)
    metadata = load_metadata(str(data_root / "HAM10000_metadata.csv"))
    train_df, val_df, test_df = patient_grouped_split(metadata, random_state=42)
    train_ds = HAM10000Dataset(train_df, str(data_root / "images"),
                               standard_transforms(train=True))
    val_loader = DataLoader(
        HAM10000Dataset(val_df, str(data_root / "images"),
                        standard_transforms(train=False)),
        batch_size=32, shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        HAM10000Dataset(test_df, str(data_root / "images"),
                        standard_transforms(train=False)),
        batch_size=32, shuffle=False, num_workers=2, pin_memory=True,
    )

    aggregator_map = {
        "FedAvg": lambda: FedAvg(),
        "AGM v6": lambda: AGMv6(),
        "History AGM": lambda: HistoryAGM(n_clients=10, beta=0.8),
        "HistAGM v8": lambda: HistAGMv8(n_clients=10, beta=0.8, warmup_rounds=5),
    }

    results = []
    for seed in args.seeds:
        for agg_name in args.aggregators:
            if agg_name not in aggregator_map:
                print(f"WARNING: Unknown aggregator '{agg_name}' — skipping.")
                continue
            for attack in args.attacks:
                key = f"seed{seed}_{agg_name.replace(' ', '_')}_{attack}"
                print(f"\n{'='*70}\n{key}\n{'='*70}")
                t0 = time.time()
                model = fed_train(
                    aggregator=aggregator_map[agg_name](),
                    attack=attack,
                    train_df=train_df, train_ds=train_ds,
                    val_loader=val_loader,
                    images_dir=str(data_root / "images"),
                    n_rounds=args.rounds, seed=seed, device=args.device,
                )
                test_acc, test_mel, _ = evaluate(
                    model, test_loader,
                    torch.device(args.device if torch.cuda.is_available() else "cpu"),
                )
                elapsed = (time.time() - t0) / 60.0
                row = {
                    "seed": seed, "aggregator": agg_name, "attack": attack,
                    "test_acc": test_acc, "test_mel": test_mel,
                    "minutes": round(elapsed, 1),
                }
                results.append(row)
                print(f"  -> acc={test_acc:.2f}% mel={test_mel:.2f}% time={elapsed:.1f}min")
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
