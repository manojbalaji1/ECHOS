"""
echos/topology.py
Manages the dynamic graph G(t) = (V, E(t)).

Edge weight update rule (Algorithm 1, line 20):
  A[i,j]^{t+1} = ρ * A[i,j]^t  +  λ * ReLU(ΔH) * Φ(X_i, X_j)

Dual-gated epistemic filter Φ (Section 2.3):
  if cosine_sim(h_i, h_j) > γ:
      Φ = exp( -||ΔW_i - ΔW_j||_F^2 / σ^2 )
  else:
      Φ = 0   # hard epistemic quarantine
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
import logging

from config import SwarmConfig

logger = logging.getLogger(__name__)


class DynamicTopology:
    """
    Maintains the N×N adjacency matrix A and computes edge updates.
    All tensors live on `device`.
    """

    def __init__(self, n_agents: int, cfg: SwarmConfig, device: str = "cpu"):
        self.n       = n_agents
        self.cfg     = cfg
        self.device  = device
        # Start with identity: each agent connected only to itself (no gossip)
        self.A       = torch.eye(n_agents, device=device)
        # History for analysis
        self._history: List[torch.Tensor] = [self.A.clone()]

    # ─────────────────────────────────────────
    # Epistemic filter  Φ(X_i, X_j)
    # ─────────────────────────────────────────

    def epistemic_filter(
        self,
        h: torch.Tensor,        # (N, d_hidden) final hidden states
        delta_W: torch.Tensor,  # (N, d_in*d_out) flattened dense LoRA deltas
        use_dual_gate: bool     = True,
        use_hard_cutoff: bool   = True,
    ) -> torch.Tensor:
        """
        Returns Φ matrix (N, N) with values in [0, 1].
        Φ[i,j] = epistemic gate from agent j → agent i.
        """
        cfg   = self.cfg
        N     = self.n

        # Cosine similarity between hidden states
        h_norm = F.normalize(h, dim=-1)                 # (N, d)
        cos_sim = h_norm @ h_norm.T                     # (N, N)

        if not use_dual_gate:
            return torch.ones(N, N, device=self.device)

        # Parameter distance (Frobenius norm squared)
        # ||ΔW_i - ΔW_j||_F^2
        diffs   = delta_W.unsqueeze(0) - delta_W.unsqueeze(1)  # (N, N, D)
        dist_sq = (diffs ** 2).sum(dim=-1)                     # (N, N)

        # RBF gate
        phi_rbf = torch.exp(-dist_sq / cfg.rbf_bandwidth)       # (N, N)

        if use_hard_cutoff:
            # Hard quarantine: Φ=0 if cosine below threshold
            mask = (cos_sim > cfg.cosine_threshold).float()
        else:
            # Soft (ablation: no_quarantine) – cosine-weighted
            mask = cos_sim.clamp(min=0)

        phi = mask * phi_rbf
        # No self-gossip
        phi.fill_diagonal_(0)
        return phi

    # ─────────────────────────────────────────
    # Topology update (Algorithm 1, line 20)
    # ─────────────────────────────────────────

    def update(
        self,
        entropy: torch.Tensor,  # (N,) current trajectory entropies
        phi: torch.Tensor,      # (N, N) epistemic filter
        use_entropy_routing: bool = True,
    ) -> None:
        """
        A^{t+1}_{ij} = ρ * A^t_{ij} + λ * ReLU(ΔH_{ij}) * Φ_{ij}
        ΔH_{ij} = (H_i - H_j) / τ   (how much more uncertain is i vs j)
        """
        cfg = self.cfg

        # ΔH matrix: ΔH[i,j] = H_i - H_j / τ
        H   = entropy.unsqueeze(1)   # (N, 1)
        dH  = (H - H.T) / cfg.temperature   # (N, N)

        if use_entropy_routing:
            routing = torch.relu(dH)         # only open edges where i is more uncertain
        else:
            # Random routing (ablation: no_entropy)
            routing = torch.rand_like(dH)

        delta_A = cfg.topology_lr * routing * phi
        self.A  = cfg.topology_decay * self.A + delta_A
        # No self-loops
        self.A.fill_diagonal_(0)
        self._history.append(self.A.clone())

    # ─────────────────────────────────────────
    # Query active edges
    # ─────────────────────────────────────────

    def active_peers(self, agent_i: int) -> List[Tuple[int, float]]:
        """
        Returns list of (peer_j, edge_weight) for agent i where weight > ε.
        """
        row  = self.A[agent_i]
        mask = row > self.cfg.edge_threshold
        idxs = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        return [(int(j), float(row[j])) for j in idxs if j != agent_i]

    def zero_edges_after_merge(self, agent_i: int) -> None:
        """Force topology decay post-merge (Algorithm 1 line 29)."""
        self.A[agent_i, :] = 0

    # ─────────────────────────────────────────
    # Analysis helpers
    # ─────────────────────────────────────────

    def adjacency_history(self) -> torch.Tensor:
        """(T, N, N) tensor of adjacency matrices over time."""
        return torch.stack(self._history, dim=0)

    def edge_formation_events(self) -> List[Dict]:
        """List of dicts recording when each edge first exceeded threshold."""
        events = []
        for t in range(1, len(self._history)):
            prev = self._history[t - 1]
            curr = self._history[t]
            newly_active = ((curr > self.cfg.edge_threshold) &
                            (prev <= self.cfg.edge_threshold))
            for i, j in newly_active.nonzero(as_tuple=False).tolist():
                if i != j:
                    events.append({"step": t, "src": j, "dst": i,
                                   "weight": float(curr[i, j])})
        return events

    def mean_in_degree(self) -> float:
        return float((self.A > self.cfg.edge_threshold).float().sum(dim=1).mean())
