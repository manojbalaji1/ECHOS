"""
experiments/scaling.py
RQ4: How does performance scale with swarm size N under strict VRAM constraints?

Varies N ∈ {3, 7, 15, 23} (capped by VRAM).
Measures:
  - Accuracy vs N  (diminishing returns?)
  - Communication overhead vs N  (validate O(K*d^2*r) vs O(N^3*L^2*d))
  - Agent specialisation (LoRA weight clustering via t-SNE / cosine sim matrix)
"""

from __future__ import annotations
import copy
import json
import logging
import os
import time
from typing import Dict, List

import numpy as np
import torch

from config import ECHOSConfig
from echos.model_loader import load_base_model
from echos.swarm import ECHOSSwarm
from echos.flops import text_debate_flops, echos_gossip_flops, measure_peak_vram, reset_vram_stats
from benchmarks.math_eval import MATHEvaluator

logger = logging.getLogger(__name__)


def run_scaling_experiment(cfg: ECHOSConfig, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer = load_base_model(cfg)
    evaluator        = MATHEvaluator(cfg, n_samples=100)

    try:
        d = model.config.hidden_size
    except AttributeError:
        d = 4096

    records: List[Dict] = []

    for n in cfg.scaling_n_agents:
        logger.info(f"\n{'='*55}\nSwarm size N = {n}\n{'='*55}")

        # Check VRAM feasibility
        if not _vram_feasible(n, d, cfg.lora.r, cfg.hardware):
            logger.warning(f"N={n} may OOM – skipping")
            continue

        scale_cfg = copy.deepcopy(cfg)
        scale_cfg.swarm.n_agents = n

        _set_seed(cfg.seed)
        reset_vram_stats(cfg.hardware.base_model_device)

        t0    = time.time()
        swarm = ECHOSSwarm(model, tokenizer, scale_cfg)
        res   = evaluator.evaluate_echos(swarm, method_name=f"echos_N{n}")
        wall  = time.time() - t0
        peak_vram = measure_peak_vram(cfg.hardware.base_model_device)

        # Analytical FLOPs
        L    = cfg.swarm.traj_window
        r    = cfg.lora.r
        K    = max(1, n // 5)
        beta = len(cfg.lora.target_modules)
        echos_f = echos_gossip_flops(n, L, d, r, K, beta)
        text_f  = text_debate_flops(n, L, d)

        # Agent specialisation: cosine similarity of adapter deltas
        specialisation_score = _compute_specialisation(swarm)

        rec = {
            "N":                    n,
            "accuracy":             res.accuracy,
            "accuracy_std":         res.std,
            "wall_time_s":          wall,
            "peak_vram_gb":         peak_vram,
            "echos_analytical_tflops": echos_f.total / 1e12,
            "text_analytical_tflops":  text_f.total  / 1e12,
            "speedup":              text_f.total / max(1, echos_f.total),
            "mean_in_degree":       swarm.topology.mean_in_degree(),
            "specialisation_score": specialisation_score,
            "n_steps_to_consensus": np.mean([
                s["step"] for s in swarm.step_outputs if _is_consensus_step(s)
            ] or [cfg.swarm.max_steps]),
        }
        records.append(rec)
        logger.info(
            f"  acc={res.accuracy:.4f}  vram={peak_vram:.2f}GB  "
            f"specialisation={specialisation_score:.4f}  "
            f"tflops={echos_f.total/1e12:.3f}"
        )

    output = {"records": records, "model": cfg.model_name}
    path   = os.path.join(output_dir, "scaling.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Scaling results → {path}")
    return output


def _compute_specialisation(swarm: ECHOSSwarm) -> float:
    """
    Measures agent specialisation as 1 - mean_cosine_similarity of adapter deltas.
    Higher = more diverse/specialised agents.
    """
    if len(swarm.agents) < 2:
        return 0.0
    try:
        flat_deltas = torch.stack([
            ag.flat_delta_concat() for ag in swarm.agents
        ])   # (N, D)
        norm = torch.nn.functional.normalize(flat_deltas, dim=-1)
        cos_mat = norm @ norm.T     # (N, N)
        # Mean off-diagonal cosine similarity
        N = cos_mat.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool)
        mean_cos = cos_mat[mask].mean().item()
        return float(1.0 - mean_cos)
    except Exception as e:
        logger.warning(f"Specialisation computation failed: {e}")
        return float("nan")


def _vram_feasible(n: int, d: int, r: int, hw) -> bool:
    """Rough VRAM feasibility check for N agents."""
    # Each LoRA adapter: ~4 target modules * 32 layers * 2 matrices * d * r * 2 bytes (bf16)
    adapter_mb_per_agent = 4 * 32 * 2 * d * r * 2 / 1e6
    total_adapter_gb     = n * adapter_mb_per_agent / 1e3
    # Conservative: assume 48GB adapter GPU
    return total_adapter_gb < 45.0


def _is_consensus_step(step_data: Dict) -> bool:
    outputs = step_data.get("outputs", {})
    answers = [v["text"].strip().lower() for v in outputs.values()]
    if not answers:
        return False
    most_common = max(set(answers), key=answers.count)
    return answers.count(most_common) / len(answers) >= 0.8


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
