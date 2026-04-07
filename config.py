"""
ECHOS Experiment Configuration
All hyperparameters, hardware settings, and experiment flags in one place.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List, Optional
import torch


# ──────────────────────────────────────────────
# Dtype helpers
# ──────────────────────────────────────────────
DTYPE_MAP = {
    "bf16":  torch.bfloat16,
    "fp16":  torch.float16,
    "fp32":  torch.float32,
}

QuantMode = Literal["fp4", "int8", "bf16", "fp16", "fp32"]


@dataclass
class HardwareConfig:
    """GPU placement and precision settings."""
    base_model_device: str = "cuda:0"          # Θbase lives here
    adapter_device: str    = "cuda:1"          # N LoRA states live here
    quant_mode: QuantMode  = "bf16"            # fp4 | int8 | bf16 | fp16 | fp32
    # bitsandbytes 4-bit sub-options (only used when quant_mode=="fp4")
    bnb_4bit_compute_dtype: str = "bf16"       # compute dtype for 4-bit ops
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"           # "nf4" | "fp4"
    max_memory: Optional[dict] = None          # e.g. {"cuda:0": "45GB", "cuda:1": "48GB"}

    def torch_dtype(self) -> torch.dtype:
        if self.quant_mode in ("fp4", "int8"):
            # quantised models still do compute in bf16 typically
            return DTYPE_MAP[self.bnb_4bit_compute_dtype]
        return DTYPE_MAP.get(self.quant_mode, torch.bfloat16)


@dataclass
class LoRAConfig:
    """LoRA adapter hyper-parameters."""
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "down_proj", "up_proj"
    ])
    lora_dropout: float = 0.0


@dataclass
class SwarmConfig:
    """ECHOS swarm algorithm parameters."""
    n_agents: int     = 15
    max_steps: int    = 20
    traj_window: int  = 32       # |T| tokens per trajectory step

    # Topology
    topology_lr: float  = 0.1   # λ  – edge update learning rate
    topology_decay: float = 0.9  # ρ  – exponential edge decay
    edge_threshold: float = 0.05 # ε  – minimum edge weight to gossip

    # Entropy routing
    temperature: float = 0.5    # τ  – entropy differential temperature

    # Dual-gated epistemic filter
    cosine_threshold: float = 0.8  # γ
    rbf_bandwidth: float    = 1.0  # σ²

    # TIES-merge
    trim_fraction: float = 0.3    # top-k% of magnitudes to keep
    merge_rate: float    = 0.5    # η  – absorption rate

    # Randomized SVD
    rsvd_oversampling: int = 10   # extra columns for sketch matrix


@dataclass
class GenerationConfig:
    """Token generation settings."""
    max_new_tokens: int   = 512
    temperature: float    = 0.7
    top_p: float          = 0.9
    do_sample: bool       = True
    repetition_penalty: float = 1.05


@dataclass
class ECHOSConfig:
    """Root config – passed everywhere."""
    # ── Model ──────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"   # any HF hub model id
    # Other good options:
    #   "meta-llama/Llama-3.1-8B-Instruct"
    #   "mistralai/Mistral-7B-Instruct-v0.3"
    #   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # ── Sub-configs ────────────────────────────
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    lora:     LoRAConfig     = field(default_factory=LoRAConfig)
    swarm:    SwarmConfig    = field(default_factory=SwarmConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # ── Experiment meta ────────────────────────
    seed: int          = 0
    seeds: List[int]   = field(default_factory=lambda: [0, 1, 2, 3, 4])
    output_dir: str    = "results"
    log_level: str     = "INFO"
    save_checkpoints: bool = False

    # ── Ablation flags (True = component enabled) ──
    use_entropy_routing: bool = True
    use_dual_gate: bool       = True
    use_epistemic_cutoff: bool = True   # hard cosine cutoff vs cosine-only
    use_ties_merge: bool      = True    # False → naive mean merge
    use_rsvd: bool            = True    # False → truncated SVD

    # ── Breakeven experiment ───────────────────
    breakeven_traj_lengths: List[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024]
    )

    # ── Scaling experiment ─────────────────────
    scaling_n_agents: List[int] = field(
        default_factory=lambda: [3, 7, 15, 23]
    )

    # ── Adversarial experiment ─────────────────
    adversarial_fraction: float = 0.2   # 20% rogue agents

    # ── Hyperparameter grid (for sweeps) ──────
    hp_grid: dict = field(default_factory=lambda: {
        "cosine_threshold": [0.7, 0.8, 0.9],
        "temperature":      [0.1, 0.5, 1.0],
        "trim_fraction":    [0.2, 0.3, 0.5],
    })


# ──────────────────────────────────────────────
# Named experiment presets
# ──────────────────────────────────────────────

def ablation_no_entropy(base: ECHOSConfig) -> ECHOSConfig:
    cfg = ECHOSConfig(**base.__dict__)
    cfg.use_entropy_routing = False
    return cfg

def ablation_no_quarantine(base: ECHOSConfig) -> ECHOSConfig:
    cfg = ECHOSConfig(**base.__dict__)
    cfg.use_epistemic_cutoff = False   # cosine-only (soft, no hard cutoff)
    return cfg

def ablation_no_epistemic(base: ECHOSConfig) -> ECHOSConfig:
    cfg = ECHOSConfig(**base.__dict__)
    cfg.use_dual_gate = False          # no gating at all
    return cfg

def ablation_naive_merge(base: ECHOSConfig) -> ECHOSConfig:
    cfg = ECHOSConfig(**base.__dict__)
    cfg.use_ties_merge = False
    return cfg

def ablation_no_svd(base: ECHOSConfig) -> ECHOSConfig:
    cfg = ECHOSConfig(**base.__dict__)
    cfg.use_rsvd = False
    return cfg


ABLATION_CONDITIONS = {
    "full_echos":      lambda c: c,
    "no_entropy":      ablation_no_entropy,
    "no_quarantine":   ablation_no_quarantine,
    "no_epistemic":    ablation_no_epistemic,
    "naive_merge":     ablation_naive_merge,
    "no_svd":          ablation_no_svd,
}
