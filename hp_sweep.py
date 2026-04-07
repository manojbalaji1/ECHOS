"""
experiments/hp_sweep.py
Hyperparameter grid search over γ, τ, k (trim fraction).
Reports best and average performance per config.
Uses MATH dataset with N=7, 2 seeds for efficiency.

Grid:
  γ (cosine_threshold): {0.7, 0.8, 0.9}
  τ (temperature):       {0.1, 0.5, 1.0}
  k (trim_fraction):     {0.2, 0.3, 0.5}
→ 27 configurations × 2 seeds = 54 runs
"""

from __future__ import annotations
import copy
import itertools
import json
import logging
import os
from typing import Dict, List

import numpy as np

from config import ECHOSConfig
from echos.model_loader import load_base_model
from echos.swarm import ECHOSSwarm
from benchmarks.math_eval import MATHEvaluator

logger = logging.getLogger(__name__)


def run_hp_sweep(cfg: ECHOSConfig, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    sweep_cfg = copy.deepcopy(cfg)
    sweep_cfg.swarm.n_agents = 7     # smaller swarm for efficiency
    sweep_seeds = sweep_cfg.seeds[:2]  # 2 seeds

    model, tokenizer = load_base_model(sweep_cfg)
    evaluator = MATHEvaluator(sweep_cfg, n_samples=150)

    grid = sweep_cfg.hp_grid
    all_keys   = sorted(grid.keys())
    all_values = [grid[k] for k in all_keys]

    records: List[Dict] = []
    best_acc = 0.0
    best_config: Dict = {}

    for combo in itertools.product(*all_values):
        hp = dict(zip(all_keys, combo))

        # Apply hyperparameters to config
        run_cfg = copy.deepcopy(sweep_cfg)
        run_cfg.swarm.cosine_threshold = hp.get("cosine_threshold", run_cfg.swarm.cosine_threshold)
        run_cfg.swarm.temperature      = hp.get("temperature",      run_cfg.swarm.temperature)
        run_cfg.swarm.trim_fraction    = hp.get("trim_fraction",    run_cfg.swarm.trim_fraction)

        seed_accs = []
        for seed in sweep_seeds:
            _set_seed(seed)
            run_cfg.seed = seed
            swarm = ECHOSSwarm(model, tokenizer, run_cfg)
            res   = evaluator.evaluate_echos(swarm, method_name="hp_sweep")
            seed_accs.append(res.accuracy)
            logger.debug(f"  {hp}  seed={seed}  acc={res.accuracy:.4f}")

        mean_acc = float(np.mean(seed_accs))
        std_acc  = float(np.std(seed_accs))
        record   = {
            **hp,
            "accuracy_mean": mean_acc,
            "accuracy_std":  std_acc,
            "accuracy_best": float(np.max(seed_accs)),
            "seed_accs":     seed_accs,
        }
        records.append(record)
        logger.info(f"  {hp}  → {mean_acc:.4f} ± {std_acc:.4f}")

        if mean_acc > best_acc:
            best_acc    = mean_acc
            best_config = hp.copy()

    output = {
        "best_config":   best_config,
        "best_accuracy": best_acc,
        "records":       records,
        "grid":          grid,
    }
    path = os.path.join(output_dir, "hp_sweep.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nBest HP: {best_config}  acc={best_acc:.4f}")
    logger.info(f"HP sweep → {path}")
    return output


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if __import__("torch").cuda.is_available():
        __import__("torch").cuda.manual_seed_all(seed)
