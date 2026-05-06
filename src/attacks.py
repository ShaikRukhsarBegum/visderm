"""
VisDerm: Byzantine attacks for federated robustness evaluation.

Implements the four attack conditions used in Section 4.6:

  - clean: No attack (honest baseline)
  - negate: Sign-flipped gradient (overt attack; Krum/Median catch this easily)
  - flip: Label-flipped local training (data poisoning; harder to detect)
  - ALIE: A Little Is Enough (Baruch et al., NeurIPS 2019) — distribution-aware
    attack that crafts updates within the honest gradient distribution

Reference: Section 4.6 of the paper.
"""
import torch


def negate_attack(update: dict, scale: float = 2.0) -> dict:
    """
    Sign-flip the update with optional amplification.

    Args:
        update: Dict mapping parameter name -> gradient tensor.
        scale: Negation magnitude (default 2.0). Higher means more disruptive.

    Returns:
        Negated update dict.
    """
    return {k: -scale * v for k, v in update.items()}


def alie_attack(honest_updates: list, std_factor: float = 1.5) -> dict:
    """
    A Little Is Enough (Baruch et al., NeurIPS 2019).

    Crafts an adversarial update that lies within the honest gradient
    distribution by displacing the mean by a small multiple of the
    coordinate-wise standard deviation. The malicious update remains
    statistically indistinguishable from honest updates, evading
    distance-based defenses.

    Args:
        honest_updates: List of honest client updates (dicts).
        std_factor: Multiplier on the coordinate-wise std (default 1.5).

    Returns:
        Adversarial update dict in the same format.
    """
    if len(honest_updates) < 2:
        raise ValueError("ALIE requires at least 2 honest updates as references")

    param_names = list(honest_updates[0].keys())
    flat = torch.stack([
        torch.cat([u[k].flatten() for k in param_names]) for u in honest_updates
    ])
    mean = flat.mean(0)
    std = flat.std(0)
    malicious_flat = mean + std_factor * std

    update = {}
    idx = 0
    for k in param_names:
        n = honest_updates[0][k].numel()
        update[k] = malicious_flat[idx:idx + n].reshape(honest_updates[0][k].shape)
        idx += n
    return update


def flip_labels(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Symmetric label-flip attack: maps class i to class (n_classes - 1 - i).

    For HAM10000 (n_classes=7) this maps each class to a different one,
    creating a consistent label-poisoning attack on a malicious client's
    local training data.
    """
    return n_classes - 1 - labels


# Convenience dispatcher used in scripts/run_federated.py
ATTACK_TYPES = {
    "clean": None,
    "negate": "negate",
    "flip": "flip",
    "alie": "alie",
}
