"""
experiments/ablations.py
Systematic ablation study (Section 4.2 of experiment design).

Tests all 6 conditions from Table 2:
  full_echos | no_entropy | no_quarantine | no_epistemic | naive_merge | no_svd

Uses MATH dataset with N=7 agents for compute efficiency.
"""

from __future__ import annotations
import copy
import json
import logging
import os
from dataclasses import replace
from typing import Dict

import numpy as np

from config import ECHOSConfig, ABLATION_CONDITIONS
from echos.model_loader import load_base_model
from echos.swarm import ECHOSSwarm
from benchmarks.math_eval import MATHEvaluator

logger = logging.getLogger(__name__)


def run_ablations(cfg: ECHOSConfig, output_dir: str) -> Dict:
    """
    Runs all ablation conditions on MATH (levels 3-5), N=7, for all seeds.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ablation config: smaller swarm, MATH only
    ablation_base = copy.deepcopy(cfg)
    ablation_base.swarm.n_agents = 7

    results: Dict = {cond: {} for cond in ABLATION_CONDITIONS}

    model, tokenizer = load_base_model(ablation_base)
    evaluator = MATHEvaluator(ablation_base, n_samples=200)

    for condition_name, modifier_fn in ABLATION_CONDITIONS.items():
        logger.info(f"\n── Ablation: {condition_name} ──")
        seed_accs = []

        for seed in ablation_base.seeds[:3]:   # 3 seeds for ablations
            _set_seed(seed)
            ablated_cfg = modifier_fn(copy.deepcopy(ablation_base))
            ablated_cfg.seed = seed

            swarm = ECHOSSwarm(model, tokenizer, ablated_cfg)
            res   = evaluator.evaluate_echos(swarm, method_name=condition_name)
            seed_accs.append(res.accuracy)

            logger.info(f"    seed={seed}  acc={res.accuracy:.4f}")

        mean_acc = float(np.mean(seed_accs))
        std_acc  = float(np.std(seed_accs))

        results[condition_name] = {
            "accuracy_mean": mean_acc,
            "accuracy_std":  std_acc,
            "seed_accs":     seed_accs,
            "config": {
                "use_entropy_routing":  ablated_cfg.use_entropy_routing,
                "use_dual_gate":        ablated_cfg.use_dual_gate,
                "use_epistemic_cutoff": ablated_cfg.use_epistemic_cutoff,
                "use_ties_merge":       ablated_cfg.use_ties_merge,
                "use_rsvd":             ablated_cfg.use_rsvd,
            }
        }
        logger.info(f"  → acc={mean_acc:.4f} ± {std_acc:.4f}")

    # Save
    path = os.path.join(output_dir, "ablations.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Ablation results → {path}")

    _print_ablation_table(results)
    return results


def _print_ablation_table(results: Dict) -> None:
    logger.info("\n\n" + "="*65)
    logger.info(f"{'Condition':<20} {'Acc Mean':>10} {'Acc Std':>10}  {'ΔAcc':>8}")
    logger.info("="*65)
    full_acc = results.get("full_echos", {}).get("accuracy_mean", 0.0)
    for cond, r in results.items():
        acc  = r.get("accuracy_mean", 0.0)
        std  = r.get("accuracy_std", 0.0)
        delta = acc - full_acc if cond != "full_echos" else 0.0
        logger.info(f"{cond:<20} {acc:>10.4f} {std:>10.4f}  {delta:>+8.4f}")
    logger.info("="*65 + "\n")


def _set_seed(seed: int) -> None:
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
