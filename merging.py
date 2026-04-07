"""
echos/merging.py
Implements:
  1. TIES-Merge  (Trimming, Electing, Sign-Resolving)
  2. Randomized SVD  rank-r projection for rank-preserving LoRA gossip
  3. Naive mean merge (ablation baseline)
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────
# 1. TIES helpers
# ─────────────────────────────────────────────────────────────────

def trim_by_magnitude(delta: torch.Tensor, top_k: float) -> torch.Tensor:
    """
    Zero out all but the top-k% largest-magnitude elements.
    delta : any shape tensor
    top_k : fraction in (0, 1]
    """
    flat    = delta.abs().flatten()
    k       = max(1, int(top_k * flat.numel()))
    thresh  = flat.kthvalue(flat.numel() - k + 1).values
    mask    = delta.abs() >= thresh
    return delta * mask


def resolve_signs(deltas: List[torch.Tensor]) -> torch.Tensor:
    """
    TIES sign election: for each parameter position choose the sign
    direction agreed upon by the majority of non-zero contributors.
    Returns a sign mask (+1 / -1) with the same shape as deltas[0].
    """
    stacked = torch.stack(deltas, dim=0)        # (K, ...)
    # Sum of signs ignoring zeros
    sign_sum = stacked.sign().sum(dim=0)
    elected  = sign_sum.sign()
    # Where no majority (sign_sum==0), fall back to +1
    elected  = torch.where(elected == 0, torch.ones_like(elected), elected)
    return elected


def ties_merge(
    deltas: List[torch.Tensor],
    top_k: float = 0.3,
    edge_weights: List[float] | None = None,
) -> torch.Tensor:
    """
    Full TIES merge of a list of LoRA dense delta matrices.

    Args:
        deltas       : list of (d_in, d_out) dense ΔW matrices
        top_k        : fraction of params to keep per delta
        edge_weights : adjacency weights (used for weighted sum)

    Returns:
        merged delta of shape (d_in, d_out)
    """
    if not deltas:
        return torch.zeros_like(deltas[0]) if deltas else None

    # Step 1 – Trim each delta
    trimmed = [trim_by_magnitude(d, top_k) for d in deltas]

    # Step 2 – Elect dominant sign per position
    elected_sign = resolve_signs(trimmed)

    # Step 3 – Keep only values whose sign matches the elected sign, then sum
    w = edge_weights if edge_weights is not None else [1.0] * len(trimmed)
    merged = torch.zeros_like(trimmed[0])
    for weight, t in zip(w, trimmed):
        aligned = t * (t.sign() == elected_sign).float()
        merged  = merged + weight * aligned

    return merged


# ─────────────────────────────────────────────────────────────────
# 2. Randomized SVD (rank-r projection)
# ─────────────────────────────────────────────────────────────────

def randomized_svd(
    M: torch.Tensor,
    rank: int,
    n_oversampling: int = 10,
    n_power_iter: int   = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD of M  ∈  R^{m × n}  →  U (m×r), S (r,), Vt (r×n)
    Following Halko et al. 2011.

    Dominant cost: sketch matrix multiply  M @ Ω  (m × (r+p) where p=n_oversampling).
    """
    device = M.device
    dtype  = M.dtype
    m, n   = M.shape
    k      = rank + n_oversampling

    # Random Gaussian sketch
    Omega = torch.randn(n, k, device=device, dtype=dtype)
    Y     = M @ Omega                               # m × k

    # Power iteration to improve accuracy
    for _ in range(n_power_iter):
        Y = M @ (M.T @ Y)

    # Orthonormal basis for range of M
    Q, _ = torch.linalg.qr(Y)                      # m × k

    # Project M into the small subspace
    B    = Q.T @ M                                  # k × n

    # Exact SVD of the small matrix
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)  # (k,r), (r,), (r,n)

    # Reconstruct full left singular vectors
    U = Q @ U_hat                                   # m × k  →  pick first r

    return U[:, :rank], S[:rank], Vt[:rank, :]


def decompose_to_lora(
    W_dense: torch.Tensor,
    rank: int,
    n_oversampling: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects a dense weight matrix back to LoRA factors A, B such that A@B ≈ W_dense.
    Returns A (d_in, r) and B (r, d_out).
    """
    U, S, Vt = randomized_svd(W_dense, rank, n_oversampling)
    # W ≈ U S Vt  → A = U * sqrt(S),  B = sqrt(S) * Vt
    sqrt_S = S.sqrt().unsqueeze(0)          # (1, r)
    A = U * sqrt_S                          # (m, r)
    B = sqrt_S.T * Vt                       # (r, n)
    return A, B


def truncated_svd_projection(
    W_dense: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ablation: exact (non-randomized) truncated SVD – much slower but deterministic."""
    U, S, Vt = torch.linalg.svd(W_dense, full_matrices=False)
    sqrt_S = S[:rank].sqrt().unsqueeze(0)
    A = U[:, :rank] * sqrt_S
    B = sqrt_S.T * Vt[:rank, :]
    return A, B


# ─────────────────────────────────────────────────────────────────
# 3. Naive mean merge (ablation)
# ─────────────────────────────────────────────────────────────────

def naive_mean_merge(
    deltas: List[torch.Tensor],
    edge_weights: List[float] | None = None,
) -> torch.Tensor:
    """Simple weighted mean – used as ablation for TIES."""
    w = edge_weights if edge_weights is not None else [1.0] * len(deltas)
    total = sum(w)
    merged = torch.zeros_like(deltas[0])
    for wi, d in zip(w, deltas):
        merged = merged + (wi / total) * d
    return merged


# ─────────────────────────────────────────────────────────────────
# 4. FLOPs for SVD steps (analytical)
# ─────────────────────────────────────────────────────────────────

def rsvd_flops(m: int, n: int, rank: int, oversampling: int = 10) -> int:
    """
    Dominant FLOPs for Randomized SVD:
      sketch:  2 * m * n * (rank + oversampling)
      QR:      2 * m * (rank + oversampling)^2
      project: 2 * (rank + oversampling) * m * n
      small SVD: O((rank + oversampling)^3)  (negligible)
    """
    k = rank + oversampling
    return 2 * m * n * k + 2 * m * k * k + 2 * k * m * n


def ties_flops(m: int, n: int, K: int) -> int:
    """
    Analytical FLOPs for TIES-merge over K peer deltas.
    Trim: O(K * m * n) magnitude comparisons (cheap, ~1 FLop each)
    Sign election: K * m * n adds
    Weighted sum:  K * m * n
    """
    return 3 * K * m * n
