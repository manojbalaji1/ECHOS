"""
echos/entropy.py
Computes per-agent trajectory entropy  H_traj(X_i) from the token-level
log-probabilities emitted during generation.

H_traj = -(1/|T|) * Σ_t Σ_v  P(x_t=v) log P(x_t=v)

We only need the full vocabulary softmax at each step to compute this,
which is obtained from the model's logits output.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (seq_len, vocab_size)  – raw pre-softmax logits for one generation.
    Returns scalar: mean per-token entropy in nats.
    """
    log_probs = F.log_softmax(logits, dim=-1)       # (T, V)
    probs     = log_probs.exp()                      # (T, V)
    # H(t) = -Σ_v p*log(p)
    per_token = -(probs * log_probs).sum(dim=-1)     # (T,)
    return per_token.mean()                           # scalar


def entropy_from_scores(
    transition_scores: torch.Tensor,
    beam_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Convenience wrapper for HuggingFace generate() with output_scores=True.
    transition_scores: tuple of (vocab_size,) tensors, one per generated token.
    Returns scalar mean entropy.
    """
    # Stack to (T, vocab_size) – already log-softmaxed by compute_transition_scores
    if isinstance(transition_scores, (list, tuple)):
        log_probs = torch.stack(transition_scores, dim=0)   # (T, V)
    else:
        log_probs = transition_scores

    probs = log_probs.exp().clamp(min=1e-12)
    per_token = -(probs * log_probs).sum(dim=-1)
    return per_token.mean()


class TrajectoryEntropyTracker:
    """
    Stateful tracker that accumulates per-step entropies for each agent
    across trajectory steps, enabling temporal analysis.
    """

    def __init__(self, n_agents: int):
        self.n_agents   = n_agents
        self.history: List[List[float]] = [[] for _ in range(n_agents)]

    def update(self, agent_idx: int, entropy_val: float) -> None:
        self.history[agent_idx].append(entropy_val)

    def latest(self, agent_idx: int) -> float:
        h = self.history[agent_idx]
        return h[-1] if h else float("inf")

    def delta(self, i: int, j: int, temperature: float = 1.0) -> float:
        """ΔH = (H_i - H_j) / τ  – positive means agent i is more uncertain."""
        return (self.latest(i) - self.latest(j)) / temperature

    def as_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Returns (N,) tensor of latest entropies."""
        return torch.tensor(
            [self.latest(i) for i in range(self.n_agents)],
            dtype=torch.float32,
            device=device,
        )

    def history_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Returns (N, T) tensor – T may vary; pads with nan."""
        max_t = max(len(h) for h in self.history)
        out = torch.full((self.n_agents, max_t), float("nan"))
        for i, h in enumerate(self.history):
            if h:
                out[i, : len(h)] = torch.tensor(h)
        return out.to(device)
