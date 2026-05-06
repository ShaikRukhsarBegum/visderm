"""
VisDerm: Training loop for Split-ViT.

Reference: Section 3.2 of the paper.
Configuration: AdamW (lr=1e-4, weight_decay=0.01), cosine annealing,
cross-entropy with label smoothing (0.1), 3× melanoma class weight,
gradient clipping at norm 1.0, 20 epochs.
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    HAM10000Dataset,
    class_weights,
    standard_transforms,
    patient_grouped_split,
    load_metadata,
    MEL_INDEX,
)
from .model import SplitViT


def evaluate(model: SplitViT, loader: DataLoader, device: torch.device):
    """
    Evaluate accuracy and per-class recall on a dataloader.

    Returns:
        accuracy_pct: Overall accuracy (%).
        mel_recall_pct: Recall on the melanoma class (%).
        per_class_recall: Tensor of per-class recall percentages.
    """
    model.eval()
    correct = 0
    total = 0
    class_tp = torch.zeros(7)
    class_total = torch.zeros(7)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            for c in range(7):
                mask = labels == c
                class_total[c] += mask.sum().item()
                class_tp[c] += ((preds == labels) & mask).sum().item()

    per_class_recall = 100.0 * class_tp / class_total.clamp(min=1)
    return (
        100.0 * correct / max(total, 1),
        per_class_recall[MEL_INDEX].item(),
        per_class_recall,
    )


def train_split_vit(
    metadata_csv: str,
    images_dir: str,
    output_dir: str,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    seed: int = 42,
    device: str = "cuda",
):
    """
    Train Split-ViT on HAM10000 with the canonical configuration.

    Args:
        metadata_csv: Path to ``HAM10000_metadata.csv``.
        images_dir: Directory containing HAM10000 images.
        output_dir: Where to save checkpoints and logs.
        epochs: Number of training epochs (default 20).
        lr: Initial learning rate (default 1e-4).
        weight_decay: AdamW weight decay (default 0.01).
        batch_size: Training batch size (default 32).
        seed: Random seed for reproducibility (default 42).
        device: Compute device.

    Returns:
        Path to the best-validation-accuracy checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. Data
    metadata = load_metadata(metadata_csv)
    train_df, val_df, test_df = patient_grouped_split(metadata, random_state=seed)

    train_loader = DataLoader(
        HAM10000Dataset(train_df, images_dir, standard_transforms(train=True)),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        HAM10000Dataset(val_df, images_dir, standard_transforms(train=False)),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        HAM10000Dataset(test_df, images_dir, standard_transforms(train=False)),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    print(f"Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # 2. Model + optimizer
    model = SplitViT().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    weights = class_weights(boost_mel_factor=3.0).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    # 3. Training loop
    best_val_acc = 0.0
    best_path = output_dir / f"model_seed{seed}.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_acc, val_mel, _ = evaluate(model, val_loader, device)
        print(
            f"E{epoch:02d}: train_loss={running_loss/n_batches:.3f} "
            f"val_acc={val_acc:.2f}% val_mel={val_mel:.2f}%"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    # 4. Final test evaluation with best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_acc, test_mel, per_class = evaluate(model, test_loader, device)
    print(f"\nFinal test (best-val checkpoint): acc={test_acc:.2f}%  mel={test_mel:.2f}%")
    print(f"Per-class recall (%): {per_class.tolist()}")
    return best_path
