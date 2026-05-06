"""
VisDerm: Federated learning aggregators.

Implements the four aggregators evaluated in Section 4.6 of the paper:

  1. FedAvg (McMahan et al., AISTATS 2017) — Simple weighted average.
  2. AGM v6 — Per-round trust scoring with iterative geometric median + cosine
     similarity weighting + outlier suppression + momentum.
  3. History AGM — Cumulative trust accumulated across rounds via
     exponential moving average (beta = 0.8 by default).
  4. HistAGM v8 — Refined History AGM with adjusted warmup behavior.

Reference: Section 4.6 (Federated Learning with Byzantine Attacks).
"""
import torch
import torch.nn.functional as F


# ============================================================
# Aggregator 1 — FedAvg
# ============================================================

class FedAvg:
    """Simple unweighted average. McMahan et al., AISTATS 2017."""
    name = "FedAvg"

    def reset(self):
        pass

    def aggregate(self, grads):
        S = torch.stack([g.flatten() for g in grads])
        return S.mean(0).reshape(grads[0].shape)


# ============================================================
# Aggregator 2 — AGM v6 (per-round trust)
# ============================================================

class AGMv6:
    """
    Per-round trust aggregation with iterative geometric median +
    cosine-similarity weighting + outlier suppression + momentum.

    No state is carried across rounds (other than the momentum term).
    See History AGM for the variant that maintains cumulative trust scores.
    """
    name = "AGM v6"

    def __init__(self, beta: float = 0.9, temperature: float = 0.1):
        self.beta = beta            # momentum decay
        self.temperature = temperature  # softmax temperature for trust weights
        self.momentum = None

    def reset(self):
        self.momentum = None

    def aggregate(self, grads):
        S = torch.stack([g.flatten() for g in grads])

        # 1. Robust estimate via 3 iterations of weighted geometric-median refinement
        est = S.median(0).values
        for _ in range(3):
            d = S - est[None]
            w = 1.0 / d.norm(dim=1, keepdim=True).clamp(min=1e-8)
            est = (w * S).sum(0) / w.sum()

        # 2. Per-client trust via cosine similarity to the robust estimate
        cs = F.cosine_similarity(S, est[None], dim=1)

        # 3. Softmax-weighted aggregation with explicit outlier zeroing
        iw = F.softmax(cs / self.temperature, dim=0)
        mu, sd = cs.mean(), cs.std()
        outlier_mask = cs < (mu - 2 * sd)
        if outlier_mask.any():
            cw = iw.clone()
            cw[outlier_mask] = 0
            result = (cw[:, None] * S).sum(0) / cw.sum().clamp(min=1e-8)
        else:
            result = (iw[:, None] * S).sum(0)

        # 4. Momentum term across rounds
        if self.momentum is None:
            self.momentum = result.clone()
        else:
            self.momentum = self.beta * self.momentum + (1 - self.beta) * result
            result = self.momentum.clone()

        return result.reshape(grads[0].shape)


# ============================================================
# Aggregator 3 — History AGM (cumulative trust)
# ============================================================

class HistoryAGM:
    """
    History-aware variant of AGM: trust is accumulated across rounds via
    exponential moving average, so persistent attackers are progressively
    down-weighted across multiple rounds rather than only the current one.

    The decay parameter ``beta`` controls how quickly past-round trust
    decays:
      - beta = 0.8 (default) means past misbehavior persists for ~5 rounds
      - beta = 1.0 means cumulative trust never decays (useful as ablation)
      - beta = 0.5 means recent rounds dominate the trust score

    Args:
        n_clients: Number of federated clients.
        beta: Trust decay parameter (default 0.8).
        temperature: Softmax temperature for converting trust → weights.
    """
    name = "History AGM"

    def __init__(self, n_clients: int, beta: float = 0.8, temperature: float = 0.1):
        self.n_clients = n_clients
        self.beta = beta
        self.temperature = temperature
        self.trust = torch.ones(n_clients)

    def reset(self):
        self.trust = torch.ones(self.n_clients)

    def aggregate(self, grads):
        S = torch.stack([g.flatten() for g in grads])
        median = S.median(0).values

        # Per-client similarity to the median
        sims = F.cosine_similarity(S, median.unsqueeze(0), dim=1).cpu()

        # Update cumulative trust via EMA
        self.trust = self.beta * self.trust + (1 - self.beta) * sims

        # Softmax-weighted aggregation by accumulated trust
        weights = F.softmax(self.trust / self.temperature, dim=0).to(S.device)
        result = (weights.unsqueeze(1) * S).sum(0)
        return result.reshape(grads[0].shape)


# ============================================================
# Aggregator 4 — HistAGM v8 (refined History AGM)
# ============================================================

class HistAGMv8(HistoryAGM):
    """
    Refined History AGM variant with adjusted warmup behavior. See Section
    4.6 of the paper for the empirical comparison against the base
    History AGM.

    The primary difference is the warmup phase: for the first
    ``warmup_rounds``, we use uniform weighting before transitioning to
    trust-weighted aggregation. This empirically reduces variance in early
    training rounds.
    """
    name = "HistAGM v8"

    def __init__(self, n_clients: int, beta: float = 0.8,
                 temperature: float = 0.1, warmup_rounds: int = 5):
        super().__init__(n_clients, beta, temperature)
        self.warmup_rounds = warmup_rounds
        self.round_count = 0

    def reset(self):
        super().reset()
        self.round_count = 0

    def aggregate(self, grads):
        self.round_count += 1
        S = torch.stack([g.flatten() for g in grads])

        if self.round_count <= self.warmup_rounds:
            # Uniform weighting during warmup
            return S.mean(0).reshape(grads[0].shape)

        return super().aggregate(grads)
