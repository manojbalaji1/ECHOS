"""
experiments/breakeven.py
RQ2: Empirical validation of the Breakeven Theorem (Section 3.3).

Varies trajectory length |T| ∈ {16, 32, 64, 128, 256, 512, 1024}
and measures:
  1. Empirical FLOPs (attention ops vs SVD ops) via torch profiler
  2. Wall-clock latency per reasoning step
  3. Accuracy at each length (does efficiency gain trade off with coherence?)

Expected: FLOPs crossover around L ≈ 488 tokens (with default params).
"""

from __future__ import annotations
import json
import logging
import os
import time
from typing import Dict, List

import torch
import numpy as np

from config import ECHOSConfig
from echos.model_loader import load_base_model
from echos.swarm import ECHOSSwarm
from echos.flops import (
    text_debate_flops, echos_gossip_flops, breakeven_L,
    EmpiricalFLOPsCounter, reset_vram_stats, measure_peak_vram,
)
from baselines.all_baselines import TextDebate
from benchmarks.math_eval import MATHEvaluator

logger = logging.getLogger(__name__)


def run_breakeven_experiment(cfg: ECHOSConfig, output_dir: str) -> Dict:
    """
    For each trajectory length in cfg.breakeven_traj_lengths, measure
    empirical FLOPs + latency for both ECHOS and text-debate.
    """
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer = load_base_model(cfg)
    evaluator        = MATHEvaluator(cfg, n_samples=50)   # small subset per L

    # Model hidden dim
    try:
        d = model.config.hidden_size
    except AttributeError:
        d = 4096

    N    = cfg.swarm.n_agents
    r    = cfg.lora.r
    K    = max(1, N // 5)
    beta = len(cfg.lora.target_modules)

    analytical_L_star = breakeven_L(d, r, K, beta)
    logger.info(f"Analytical breakeven L* = {analytical_L_star:.1f} tokens")

    records: List[Dict] = []

    for traj_len in cfg.breakeven_traj_lengths:
        logger.info(f"\n── Trajectory length |T| = {traj_len} ──")

        # ── ECHOS empirical ─────────────────────────────────────
        echos_cfg = _clone_cfg(cfg)
        echos_cfg.swarm.traj_window     = traj_len
        echos_cfg.generation.max_new_tokens = traj_len

        reset_vram_stats(cfg.hardware.base_model_device)
        t0 = time.time()

        with EmpiricalFLOPsCounter(model) as echos_counter:
            swarm = ECHOSSwarm(model, tokenizer, echos_cfg)
            echos_res = evaluator.evaluate_echos(swarm, method_name="echos")

        echos_time  = time.time() - t0
        echos_flops = float(echos_counter.total_flops)
        echos_vram  = measure_peak_vram(cfg.hardware.base_model_device)

        logger.info(
            f"  ECHOS:  flops={echos_flops/1e12:.3f}T  "
            f"time={echos_time:.1f}s  acc={echos_res.accuracy:.4f}  "
            f"vram={echos_vram:.2f}GB"
        )

        # ── Text Debate empirical ────────────────────────────────
        text_cfg = _clone_cfg(cfg)
        text_cfg.generation.max_new_tokens = traj_len
        text_debate = TextDebate(
            model, tokenizer, text_cfg,
            n_agents=N,
            max_context_tokens=min(traj_len * N, 4096),
        )

        reset_vram_stats(cfg.hardware.base_model_device)
        t0 = time.time()

        with EmpiricalFLOPsCounter(model) as text_counter:
            td_accs = []
            for sample in evaluator.dataset:
                prompt  = evaluator.format_prompt(sample)
                raw     = text_debate.solve(prompt)
                pred    = evaluator.extract_answer(raw)
                td_accs.append(evaluator.is_correct(pred, sample["answer"]))

        text_time  = time.time() - t0
        text_flops = float(text_counter.total_flops)
        text_vram  = measure_peak_vram(cfg.hardware.base_model_device)
        text_acc   = float(np.mean(td_accs))

        logger.info(
            f"  Text:   flops={text_flops/1e12:.3f}T  "
            f"time={text_time:.1f}s  acc={text_acc:.4f}  "
            f"vram={text_vram:.2f}GB"
        )

        # ── Analytical FLOPs ─────────────────────────────────────
        analy_text  = text_debate_flops(N, traj_len, d)
        analy_echos = echos_gossip_flops(N, traj_len, d, r, K, beta)

        records.append({
            "traj_length":       traj_len,
            # Empirical
            "echos_empirical_tflops":   echos_flops / 1e12,
            "text_empirical_tflops":    text_flops  / 1e12,
            "echos_wall_time_s":        echos_time,
            "text_wall_time_s":         text_time,
            "echos_accuracy":           echos_res.accuracy,
            "text_accuracy":            text_acc,
            "echos_peak_vram_gb":       echos_vram,
            "text_peak_vram_gb":        text_vram,
            # Analytical
            "echos_analytical_tflops":  analy_echos.total / 1e12,
            "text_analytical_tflops":   analy_text.total  / 1e12,
            "analytical_speedup":       analy_text.total / max(1, analy_echos.total),
            "empirical_speedup":        text_flops / max(1, echos_flops),
        })

    # Detect empirical crossover
    crossover_L = _find_empirical_crossover(records)
    logger.info(f"\nEmpirical crossover at L ≈ {crossover_L}  (analytical: {analytical_L_star:.1f})")

    output = {
        "analytical_breakeven_L": analytical_L_star,
        "empirical_breakeven_L":  crossover_L,
        "model_d":                d,
        "N": N, "r": r, "K": K, "beta": beta,
        "records":                records,
    }
    path = os.path.join(output_dir, "breakeven.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Breakeven results → {path}")
    return output


def _find_empirical_crossover(records: List[Dict]) -> float:
    """Find L where ECHOS becomes cheaper than text-debate empirically."""
    for i in range(len(records) - 1):
        a, b = records[i], records[i + 1]
        if (a["echos_empirical_tflops"] > a["text_empirical_tflops"]) and \
           (b["echos_empirical_tflops"] <= b["text_empirical_tflops"]):
            # Interpolate
            La, Lb = a["traj_length"], b["traj_length"]
            return float((La + Lb) / 2)
    return float("nan")


def _clone_cfg(cfg: ECHOSConfig) -> ECHOSConfig:
    import copy
    return copy.deepcopy(cfg)
