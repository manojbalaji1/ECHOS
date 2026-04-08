"""
experiments/main_comparison.py
RQ1 + RQ2: Main accuracy and efficiency comparison on all benchmarks.

Produces Table 1 of the paper:
  Rows:    Single-Agent CoT | Text Debate | Sparse Text |
           Static LoRA Avg  | LoRA Ensemble | ECHOS (ours)
  Columns: MATH | GPQA | StrategyQA | SWE-bench Lite | TFLOPs | VRAM
"""

from __future__ import annotations
import json
import logging
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch

from config import ECHOSConfig
from echos.model_loader import load_base_model
from echos.swarm import ECHOSSwarm
from echos.flops import (
    echos_gossip_flops,
    text_debate_flops, echos_gossip_flops as echos_flops,
)
from baselines.all_baselines import (
    SingleAgentCoT, TextDebate, SparseText,
    StaticLoRAAverage, IndividualLoRAEnsemble,
)
from benchmarks.math_eval import MATHEvaluator
from benchmarks.gpqa_eval import GPQAEvaluator
from benchmarks.strategy_qa import StrategyQAEvaluator
from benchmarks.swe_bench_eval import SWEBenchEvaluator
from echos.flops import text_debate_flops, echos_gossip_flops

logger = logging.getLogger(__name__)

# ── Safe imports from flops module ──
try:
    from echos.flops import text_debate_flops, echos_gossip_flops
except ImportError:
    pass


def run_main_comparison(cfg: ECHOSConfig, output_dir: str) -> Dict:
    """
    Full main comparison across all methods and benchmarks.
    Results saved incrementally to output_dir/main_comparison.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # ── Load model ──────────────────────────────────────────────
    model, tokenizer = load_base_model(cfg)

    # ── Benchmark evaluators ─────────────────────────────────────
    evaluators = {
        "math":        MATHEvaluator(cfg, n_samples=500),
        "gpqa":        GPQAEvaluator(cfg, n_samples=198),
        "strategy_qa": StrategyQAEvaluator(cfg, n_samples=500),
        "swe_bench":   SWEBenchEvaluator(cfg, n_samples=50),
    }

    # ── Build ECHOS swarm ────────────────────────────────────────
    swarm = ECHOSSwarm(model, tokenizer, cfg)

    # ── Build baselines ──────────────────────────────────────────
    baselines = {
        "single_agent_greedy": SingleAgentCoT(model, tokenizer, cfg, n_samples=1),
        "self_consistency_64": SingleAgentCoT(model, tokenizer, cfg, n_samples=64),
        "text_debate":         TextDebate(model, tokenizer, cfg,
                                          n_agents=cfg.swarm.n_agents),
        "sparse_text":         SparseText(model, tokenizer, cfg,
                                          n_agents=cfg.swarm.n_agents),
        "static_lora_avg":     StaticLoRAAverage(model, tokenizer, cfg),
        "lora_ensemble":       IndividualLoRAEnsemble(model, tokenizer, cfg),
    }

    # ── Run experiments ──────────────────────────────────────────
    for bench_name, evaluator in evaluators.items():
        results[bench_name] = {}
        logger.info(f"\n{'='*60}\nBenchmark: {bench_name.upper()}\n{'='*60}")

        # ECHOS
        logger.info("Running ECHOS...")
        t0 = time.time()
        echos_res = evaluator.evaluate_echos(swarm, method_name="echos")
        echos_res.meta["wall_time_s"] = time.time() - t0
        echos_res.meta["total_tflops"] = swarm.flops_log.total_tflops()
        echos_res.meta["peak_vram_gb"] = swarm.flops_log.peak_vram_gb()
        results[bench_name]["echos"] = echos_res.to_dict()

        # Baselines
        for bl_name, baseline in baselines.items():
            logger.info(f"Running {bl_name}...")
            t0 = time.time()
            if hasattr(baseline, "solve"):
                bl_results = []
                for sample in evaluator.dataset:
                    prompt  = evaluator.format_prompt(sample)
                    raw     = baseline.solve(prompt)
                    pred    = evaluator.extract_answer(raw)
                    correct = evaluator.is_correct(pred, sample["answer"])
                    bl_results.append(correct)
                acc = float(np.mean(bl_results))
                boots = [
                    np.mean(random.choices(bl_results, k=len(bl_results)))
                    for _ in range(1000)
                ]
                std = float(np.std(boots))
            else:
                acc, std = 0.0, 0.0

            results[bench_name][bl_name] = {
                "accuracy": acc,
                "std":      std,
                "wall_time_s": time.time() - t0,
            }

        # Save incrementally
        out_path = os.path.join(output_dir, "main_comparison.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved → {out_path}")

    # ── Analytical FLOPs table ───────────────────────────────────
    _save_flops_table(cfg, output_dir)

    return results


def _save_flops_table(cfg: ECHOSConfig, output_dir: str) -> None:
    """
    Compute analytical FLOPs for text-debate vs ECHOS across trajectory lengths.
    """
    from echos.flops import text_debate_flops, echos_gossip_flops, breakeven_L
    import transformers

    # Get model hidden dim from config (look up from common models)
    d_map = {
        "7b": 4096, "8b": 4096, "13b": 5120, "70b": 8192,
    }
    model_lower = cfg.model_name.lower()
    d = next((v for k, v in d_map.items() if k in model_lower), 4096)

    N       = cfg.swarm.n_agents
    r       = cfg.lora.r
    K       = max(1, N // 5)   # approximate average in-degree
    beta    = len(cfg.lora.target_modules)
    n_layers = 32

    table = []
    for L in cfg.breakeven_traj_lengths:
        text_f  = text_debate_flops(N, L, d, n_layers)
        echos_f = echos_gossip_flops(N, L, d, r, K, beta, n_layers)
        table.append({
            "L":             L,
            "text_tflops":  text_f.total / 1e12,
            "echos_tflops": echos_f.total / 1e12,
            "speedup":      text_f.total / max(1, echos_f.total),
        })

    L_star = breakeven_L(d, r, K, beta)
    flops_table = {
        "breakeven_L_analytical": L_star,
        "N": N, "r": r, "K": K, "d": d,
        "entries": table,
    }
    path = os.path.join(output_dir, "flops_table.json")
    with open(path, "w") as f:
        json.dump(flops_table, f, indent=2)
    logger.info(f"FLOPs table → {path}  (L* ≈ {L_star:.1f})")
