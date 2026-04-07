"""
echos/flops.py
FLOPs measurement utilities.

Two modes:
  1. Analytical  – closed-form formulas from Section 3 of the paper.
  2. Empirical   – torch.profiler-based measurement during real runs.

The breakeven theorem is validated by comparing both modes.
"""

from __future__ import annotations
import math
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
try:
    from torch.utils.flop_counter import FlopCounterMode
    _HAS_FLOP_COUNTER = True
except ImportError:
    _HAS_FLOP_COUNTER = False

from echos.merging import rsvd_flops, ties_flops

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Analytical FLOPs
# ─────────────────────────────────────────────────────────────────

@dataclass
class AnalyticalFLOPs:
    """All FLOPs counts are in floating-point operations (multiply-add = 2 FLOPs)."""
    method: str
    attention_flops: int = 0
    svd_flops: int       = 0
    ties_flops: int      = 0
    dense_delta_flops: int = 0
    total: int           = 0

    def __post_init__(self):
        self.total = (self.attention_flops + self.svd_flops +
                      self.ties_flops + self.dense_delta_flops)


def text_debate_flops(
    N: int,      # number of agents
    L: int,      # trajectory length (tokens)
    d: int,      # model hidden dimension
    n_layers: int = 32,
) -> AnalyticalFLOPs:
    """
    Standard text-based broadcasting (Section 3.1).
    Context = N*L tokens. Per-layer attention = 4*(N*L)^2*d FLOPs.
    Total across swarm = N * n_layers * 4 * (N*L)^2 * d
    """
    per_layer = 4 * (N * L) ** 2 * d
    total_attn = N * n_layers * per_layer
    return AnalyticalFLOPs(
        method="text_debate",
        attention_flops=total_attn,
        total=total_attn,
    )


def echos_gossip_flops(
    N: int,
    L: int,      # trajectory length (informational only – ECHOS doesn't depend on L)
    d: int,
    r: int,
    K: int,           # average in-degree (peers per agent)
    beta: int = 4,    # number of LoRA target matrices per layer
    n_layers: int = 32,
    oversampling: int = 10,
) -> AnalyticalFLOPs:
    """
    ECHOS parameter gossip (Section 3.2).
    Per agent:
      Dense delta construction: K * (2 * d^2 * r)  per layer
      TIES merge:               O(K * d^2)          per layer
      Randomized SVD:           4 * d^2 * r          per layer (dominant)
    Total across all agents and layers: N * beta * n_layers * sum_above
    """
    per_layer_dense  = K * 2 * d * d * r             # K dense deltas
    per_layer_ties   = ties_flops(d, d, K)            # TIES
    per_layer_rsvd   = rsvd_flops(d, d, r, oversampling)

    per_agent = beta * n_layers * (per_layer_dense + per_layer_ties + per_layer_rsvd)
    total     = N * per_agent

    return AnalyticalFLOPs(
        method="echos_gossip",
        dense_delta_flops = N * beta * n_layers * per_layer_dense,
        ties_flops        = N * beta * n_layers * per_layer_ties,
        svd_flops         = N * beta * n_layers * per_layer_rsvd,
        total             = total,
    )


def breakeven_L(
    d: int,
    r: int,
    K: int,
    beta: int = 4,
    oversampling: int = 10,
) -> float:
    """
    Solves for L* where FLOPs_text = FLOPs_echos.
    From Section 3.3:
      L > sqrt( beta * d * r / (4*K) )
    Returns L* as a float.
    """
    return math.sqrt(beta * d * r / (4 * K))


# ─────────────────────────────────────────────────────────────────
# Empirical FLOPs via torch.profiler
# ─────────────────────────────────────────────────────────────────

class EmpiricalFLOPsCounter:
    """
    Context manager that measures actual FLOPs for a code block.
    Usage:
        with EmpiricalFLOPsCounter(model) as counter:
            model(input_ids=...)
        print(counter.total_flops)
    """

    def __init__(self, model: Optional[torch.nn.Module] = None):
        self.model       = model
        self.total_flops = 0
        self._ctx        = None

    def __enter__(self):
        if _HAS_FLOP_COUNTER and self.model is not None:
            self._ctx = FlopCounterMode(self.model, display=False)
            self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self._ctx is not None:
            self._ctx.__exit__(*args)
            self.total_flops = self._ctx.get_total_flops()
        return False


@dataclass
class StepFLOPs:
    """Recorded FLOPs for one reasoning step."""
    step: int
    traj_length: int
    n_agents: int
    method: str
    generation_flops: float  = 0.0
    gossip_flops: float      = 0.0
    total_flops: float       = 0.0
    wall_time_s: float       = 0.0
    peak_vram_gb: float      = 0.0

    def flops_per_token(self) -> float:
        denom = max(1, self.traj_length * self.n_agents)
        return self.total_flops / denom


@dataclass
class FLOPsLog:
    """Accumulates FLOPs records across an experiment run."""
    records: list = field(default_factory=list)

    def append(self, rec: StepFLOPs):
        self.records.append(rec)

    def total_tflops(self) -> float:
        return sum(r.total_flops for r in self.records) / 1e12

    def total_wall_time(self) -> float:
        return sum(r.wall_time_s for r in self.records)

    def peak_vram_gb(self) -> float:
        return max((r.peak_vram_gb for r in self.records), default=0.0)

    def to_dict_list(self) -> list:
        return [vars(r) for r in self.records]


def measure_peak_vram(device: str = "cuda:0") -> float:
    """Returns current peak VRAM in GB for a device."""
    if not torch.cuda.is_available():
        return 0.0
    idx = int(device.split(":")[-1]) if ":" in device else 0
    return torch.cuda.max_memory_allocated(idx) / (1024 ** 3)


def reset_vram_stats(device: str = "cuda:0") -> None:
    if torch.cuda.is_available():
        idx = int(device.split(":")[-1]) if ":" in device else 0
        torch.cuda.reset_peak_memory_stats(idx)
