"""
run_experiments.py
Master CLI for the full ECHOS TMLR experiment suite.

Usage examples
--------------
# Full suite (all experiments, default model, bf16):
python run_experiments.py --all

# Quick smoke-test (N=3, 10 samples):
python run_experiments.py --smoke

# Single experiment:
python run_experiments.py --exp main_comparison

# Custom model + quantization:
python run_experiments.py --all \\
    --model "meta-llama/Llama-3.1-8B-Instruct" \\
    --quant fp4

# Ablations only:
python run_experiments.py --exp ablations --n_agents 7 --n_samples 200
"""

from __future__ import annotations
import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

# ── Project root on path ───────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import ECHOSConfig, HardwareConfig, SwarmConfig, LoRAConfig, GenerationConfig
from echos.model_loader import load_base_model
from echos.swarm import ECHOSSwarm
from experiments.main_comparison import run_main_comparison
from experiments.ablations import run_ablations
from experiments.breakeven import run_breakeven_experiment
from experiments.scaling import run_scaling_experiment
from experiments.mechanistic import (
    run_entropy_topology_correlation,
    run_adversarial_attack,
    run_tangent_space_verification,
)
from experiments.hp_sweep import run_hp_sweep
from analysis.plots import generate_all_figures
from analysis.stats import save_all_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("echos.runner")


# ─────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ECHOS TMLR Experiment Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Model ───────────────────────────────────────────────────
    p.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)"
    )
    p.add_argument(
        "--quant", default="bf16",
        choices=["fp4", "int8", "bf16", "fp16", "fp32"],
        help="Quantization mode (default: bf16)"
    )

    # ── GPU placement ───────────────────────────────────────────
    p.add_argument("--base-gpu",    default="cuda:0", help="GPU for base model")
    p.add_argument("--adapter-gpu", default="cuda:1", help="GPU for LoRA adapters")

    # ── Experiment selection ────────────────────────────────────
    p.add_argument("--all",   action="store_true", help="Run full experiment suite")
    p.add_argument("--smoke", action="store_true", help="Smoke test (tiny N, few samples)")
    p.add_argument(
        "--exp", nargs="+",
        choices=[
            "main_comparison", "ablations", "breakeven",
            "scaling", "mechanistic", "adversarial",
            "tangent_space", "hp_sweep", "plots", "stats",
        ],
        help="Specific experiments to run"
    )

    # ── Swarm config overrides ───────────────────────────────────
    p.add_argument("--n_agents",   type=int, default=None, help="Override N agents")
    p.add_argument("--n_samples",  type=int, default=None, help="Override benchmark sample count")
    p.add_argument("--max_steps",  type=int, default=None, help="Override max swarm steps")
    p.add_argument("--lora_rank",  type=int, default=8,    help="LoRA rank r (default: 8)")
    p.add_argument("--traj_window", type=int, default=32,  help="Trajectory window |T| tokens")

    # ── Output ──────────────────────────────────────────────────
    p.add_argument("--output_dir", default="results", help="Root output directory")
    p.add_argument("--seed",       type=int, default=0, help="Random seed")

    # ── Misc ────────────────────────────────────────────────────
    p.add_argument("--debug", action="store_true", help="Set log level to DEBUG")
    p.add_argument("--skip_swe", action="store_true",
                   help="Skip SWE-bench (requires Docker + harness setup)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────

def build_config(args: argparse.Namespace) -> ECHOSConfig:
    cfg = ECHOSConfig(
        model_name  = args.model,
        output_dir  = args.output_dir,
        seed        = args.seed,
        hardware    = HardwareConfig(
            base_model_device = args.base_gpu,
            adapter_device    = args.adapter_gpu,
            quant_mode        = args.quant,
        ),
        lora = LoRAConfig(r=args.lora_rank),
        swarm = SwarmConfig(
            n_agents    = args.n_agents   or 15,
            max_steps   = args.max_steps  or 20,
            traj_window = args.traj_window,
        ),
    )

    if args.smoke:
        cfg.swarm.n_agents   = 3
        cfg.swarm.max_steps  = 3
        cfg.seeds            = [0]

    return cfg


# ─────────────────────────────────────────────────────────────────
# Experiment dispatch
# ─────────────────────────────────────────────────────────────────

def run_suite(cfg: ECHOSConfig, exps: List[str], args: argparse.Namespace) -> None:
    out   = cfg.output_dir
    n_smp = args.n_samples

    timings: dict = {}

    def _time(fn, *a, **kw):
        t0 = time.time()
        result = fn(*a, **kw)
        timings[fn.__name__] = time.time() - t0
        return result

    if "main_comparison" in exps:
        logger.info("\n" + "█"*60 + "\n  MAIN COMPARISON\n" + "█"*60)
        main_cfg = copy.deepcopy(cfg)
        if n_smp:
            # Monkey-patch evaluator n_samples via cfg meta
            main_cfg.__dict__["_n_samples_override"] = n_smp
        _time(run_main_comparison, main_cfg, out)

    if "ablations" in exps:
        logger.info("\n" + "█"*60 + "\n  ABLATION STUDY\n" + "█"*60)
        abl_cfg = copy.deepcopy(cfg)
        abl_cfg.swarm.n_agents = 7
        _time(run_ablations, abl_cfg, out)

    if "breakeven" in exps:
        logger.info("\n" + "█"*60 + "\n  BREAKEVEN THEOREM\n" + "█"*60)
        _time(run_breakeven_experiment, copy.deepcopy(cfg), out)

    if "scaling" in exps:
        logger.info("\n" + "█"*60 + "\n  SCALING ANALYSIS\n" + "█"*60)
        _time(run_scaling_experiment, copy.deepcopy(cfg), out)

    if "mechanistic" in exps:
        logger.info("\n" + "█"*60 + "\n  MECHANISTIC: ENTROPY-TOPOLOGY\n" + "█"*60)
        _time(run_entropy_topology_correlation, copy.deepcopy(cfg), out)

    if "adversarial" in exps:
        logger.info("\n" + "█"*60 + "\n  MECHANISTIC: ADVERSARIAL ATTACK\n" + "█"*60)
        _time(run_adversarial_attack, copy.deepcopy(cfg), out)

    if "tangent_space" in exps:
        logger.info("\n" + "█"*60 + "\n  MECHANISTIC: TANGENT SPACE\n" + "█"*60)
        _time(run_tangent_space_verification, copy.deepcopy(cfg), out)

    if "hp_sweep" in exps:
        logger.info("\n" + "█"*60 + "\n  HYPERPARAMETER SWEEP\n" + "█"*60)
        _time(run_hp_sweep, copy.deepcopy(cfg), out)

    if "plots" in exps:
        logger.info("\n" + "█"*60 + "\n  GENERATING FIGURES\n" + "█"*60)
        figs_dir = os.path.join(out, "figures")
        generate_all_figures(out, figs_dir)

    if "stats" in exps:
        logger.info("\n" + "█"*60 + "\n  STATISTICAL ANALYSIS\n" + "█"*60)
        stats_dir = os.path.join(out, "stats")
        save_all_stats(out, stats_dir)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUITE COMPLETE")
    logger.info("="*60)
    for name, t in timings.items():
        logger.info(f"  {name:<35} {t/60:.1f} min")
    logger.info(f"  {'Total':<35} {sum(timings.values())/60:.1f} min")
    logger.info(f"\n  Results saved to: {os.path.abspath(out)}")
    logger.info("="*60)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Seed everything globally
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        logger.info(
            f"GPUs: "
            + ", ".join(
                f"{torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory // 1024**3}GB)"
                for i in range(torch.cuda.device_count())
            )
        )

    cfg = build_config(args)
    logger.info(f"Model:  {cfg.model_name}")
    logger.info(f"Quant:  {cfg.hardware.quant_mode}")
    logger.info(f"N:      {cfg.swarm.n_agents}  r={cfg.lora.r}  |T|={cfg.swarm.traj_window}")
    logger.info(f"Output: {os.path.abspath(cfg.output_dir)}")

    # Determine which experiments to run
    if args.smoke:
        exps = ["main_comparison", "ablations", "breakeven", "plots"]
        logger.info("Smoke mode: running subset of experiments with tiny settings")
    elif args.all:
        exps = [
            "main_comparison", "ablations", "breakeven",
            "scaling", "mechanistic", "adversarial",
            "tangent_space", "hp_sweep", "plots", "stats",
        ]
    elif args.exp:
        exps = args.exp
    else:
        logger.error("Must specify --all, --smoke, or --exp <name>")
        sys.exit(1)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump({k: str(v) for k, v in vars(cfg).items()}, f, indent=2)

    run_suite(cfg, exps, args)


if __name__ == "__main__":
    main()
