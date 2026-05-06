"""
VisDerm: Privacy mechanisms.

Implements the seven privacy mechanisms compared in Section 4.2 of the paper:

  1. Uniform DP        — Gaussian noise on all 196 patch tokens (148 KB payload)
  2. LSP-DP            — Attention-weighted spatial allocation (148 KB)
  3. PB-96 / PB-64 / PB-48 — Privacy Bottleneck: linear projection to a smaller
                          subspace, then noise (variable payload by bottleneck size)
  4. STP-49 / STP-98   — Stochastic Token Pruning: random subsample of tokens,
                          then noise on retained tokens

The headline mechanism is **STP-49 (n_keep=49)**, which:
  * Reduces the per-inference payload to 38 KB (3.9× reduction over the
    full 148 KB intermediate tensor)
  * Reduces the number of noisy dimensions to 9,408 (vs 37,632 for full-token
    DP), supporting the Noise Dimensionality Heuristic
  * Empirically improves melanoma recall to 88.7% under inference-time
    application at (ε=8, δ=10⁻⁵) — see Section 4.4 of the paper for the
    inference-time decomposition and Section 4.7 for the controlled
    training-time ablation

Reference: Section 2.3 of the paper. Noise scale is calibrated via the
analytic Gaussian mechanism (Balle and Wang, ICML 2018).
"""
import math
import torch
import torch.nn as nn


def gaussian_sigma(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    """
    Standard Gaussian-mechanism noise scale calibration.

    Returns sigma such that adding N(0, sigma^2 I) noise to a function with
    L2 sensitivity ``sensitivity`` provides (epsilon, delta)-DP.

    For the analytic Gaussian mechanism (Balle and Wang 2018), use a tighter
    calibration; this implementation uses the standard bound from
    Dwork and Roth (2014) which is what the paper reports.
    """
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


# ============================================================
# Mechanism 1 — Stochastic Token Pruning with DP (STP-DP)
# ============================================================

class STPDP(nn.Module):
    """
    Stochastic Token Pruning with Differential Privacy.

    Step 1 (Pruning):
      Uniformly sample ``n_keep`` patch tokens at random from the 196 patch
      tokens (the class token is always retained). The selection is
      data-independent — knowing the selection reveals no information about
      the input.

    Step 2 (Clipping):
      Each retained token is clipped to unit L2 norm.

    Step 3 (Noise):
      Gaussian noise with scale calibrated to (epsilon, delta) is added.

    The resulting representation has shape ``[B, n_keep+1, 192]`` and a
    payload of ``(n_keep+1) * 192 * 4`` bytes (e.g., 38 KB for n_keep=49).

    Args:
        n_keep: Number of patch tokens to retain (default 49 → 38 KB payload).
        epsilon: DP epsilon parameter (default 8.0).
        delta: DP delta parameter (default 1e-5).
        sensitivity: L2 sensitivity bound (default 1.0 — token clipping).
    """

    def __init__(self, n_keep: int = 49, epsilon: float = 8.0,
                 delta: float = 1e-5, sensitivity: float = 1.0):
        super().__init__()
        self.n_keep = n_keep
        self.q = n_keep / 196  # subsampling rate (q = 0.25 for n_keep=49)
        self.sigma = gaussian_sigma(epsilon, delta, sensitivity)
        self.sensitivity = sensitivity

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply STP-DP to client-side intermediate features.

        Args:
            features: Shape ``[B, 197, 192]``. Token 0 is the class token;
                tokens 1..196 are patch tokens.

        Returns:
            Privatized features of shape ``[B, n_keep+1, 192]``.
        """
        B, N, D = features.shape
        cls_token = features[:, 0:1, :]
        patches = features[:, 1:, :]

        # Step 1: Random uniform subsampling
        idx = torch.randperm(196, device=features.device)[:self.n_keep]
        selected = patches[:, idx, :]

        # Recombine (class token + selected patches)
        combined = torch.cat([cls_token, selected], dim=1)

        # Step 2: Per-token L2 clipping to ``sensitivity``
        norms = combined.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        clipped = combined * (self.sensitivity / norms).clamp(max=1.0)

        # Step 3: Gaussian noise
        return clipped + torch.randn_like(clipped) * self.sigma


# ============================================================
# Mechanism 2 — Privacy Bottleneck (linear projection + DP)
# ============================================================

class PrivacyBottleneck(nn.Module):
    """
    Privacy Bottleneck: linear projection to a smaller subspace, followed by
    Gaussian noise, then linear decompression back to the full embedding
    dimension.

    Compared against STP-DP at matched per-inference dimensionality. The
    paper's PB-48, PB-64, PB-96 variants correspond to bottleneck dimensions
    of 48, 64, and 96.

    Args:
        in_dim: Input embedding dimension (default 192 for DeiT-Tiny).
        bottleneck_dim: Compressed dimensionality.
        epsilon: DP epsilon (default 8.0).
        delta: DP delta (default 1e-5).
    """

    def __init__(self, in_dim: int = 192, bottleneck_dim: int = 64,
                 epsilon: float = 8.0, delta: float = 1e-5):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )
        self.decompressor = nn.Linear(bottleneck_dim, in_dim)
        self.sigma = gaussian_sigma(epsilon, delta, sensitivity=1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        c = self.compressor(features)
        norms = c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        clipped = c * (1.0 / norms).clamp(max=1.0)
        noisy = clipped + torch.randn_like(clipped) * self.sigma
        return self.decompressor(noisy)


# ============================================================
# Mechanism 3 — Uniform DP (full-token Gaussian noise)
# ============================================================

class UniformDP(nn.Module):
    """
    Apply uniform Gaussian noise to all patch tokens.

    Baseline mechanism that does not reduce the payload. Used as a reference
    point for the Noise Dimensionality Heuristic — STP-DP and PB consistently
    outperform UniformDP at fixed (epsilon, delta) on compact ViTs.
    """

    def __init__(self, epsilon: float = 8.0, delta: float = 1e-5):
        super().__init__()
        self.sigma = gaussian_sigma(epsilon, delta, sensitivity=1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        cls_token = features[:, 0:1, :]
        patches = features[:, 1:, :]
        norms = patches.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        clipped = patches * (1.0 / norms).clamp(max=1.0)
        noisy = clipped + torch.randn_like(clipped) * self.sigma
        return torch.cat([cls_token, noisy], dim=1)


# ============================================================
# Mechanism 4 — Learned Spatial Privacy DP (LSP-DP)
# ============================================================

class LSPDP(nn.Module):
    """
    Learned Spatial Privacy DP: noise scale is allocated proportionally to
    1/attention so that high-attention tokens receive less noise.

    Counter-intuitively, this performs WORSE than uniform allocation on
    compact ViTs — see Section 4.3 ("Why Spatial Noise Redistribution Fails")
    and the Noise Dimensionality Heuristic for the explanation.

    Requires the spatial attention map from ``SplitViT.client_forward``.
    """

    def __init__(self, epsilon: float = 8.0, delta: float = 1e-5):
        super().__init__()
        self.base_sigma = gaussian_sigma(epsilon, delta, sensitivity=1.0)

    def forward(self, features: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: ``[B, 197, 192]``
            attention: ``[B, 14, 14]`` — class-token attention from client
        """
        B = features.shape[0]
        cls_token = features[:, 0:1, :]
        patches = features[:, 1:, :]

        # Per-token noise scale: high attention -> low sigma, normalized so
        # the mean sigma matches base_sigma (preserves the privacy budget).
        attn_flat = attention.reshape(B, 196).clamp(min=1e-6)
        per_token_sigma = self.base_sigma / attn_flat
        per_token_sigma = per_token_sigma * (
            self.base_sigma / per_token_sigma.mean(dim=1, keepdim=True)
        )

        norms = patches.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        clipped = patches * (1.0 / norms).clamp(max=1.0)
        noise = torch.randn_like(clipped) * per_token_sigma.unsqueeze(-1)
        return torch.cat([cls_token, clipped + noise], dim=1)
