"""
experiments/mechanistic.py
RQ3 + RQ5: Mechanistic validation experiments.

5.1  Entropy-Topology Correlation
       - Mutual information between ΔH and edge formation
       - Heatmap of A^(t) over time
       - Entropy curves (should spike before consensus, then drop)

5.2  Epistemic Filter Efficacy
       - 3/15 adversarial agents (fine-tuned on "always answer 42")
       - Contamination rate, quarantine success, time-to-isolation

5.3  Tangent Space Verification
       - Cosine similarity of ΔW_i and ΔW_j before/after TIES vs naive merge
"""

from __future__ import annotations
import copy
import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from config import ECHOSConfig
from echos.model_loader import load_base_model
from echos.swarm import ECHOSSwarm
from echos.merging import ties_merge, naive_mean_merge
from benchmarks.math_eval import MATHEvaluator
from benchmarks.strategy_qa import AdversarialMathQAEvaluator

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# 5.1  Entropy-Topology Correlation
# ═════════════════════════════════════════════════════════════════

def run_entropy_topology_correlation(cfg: ECHOSConfig, output_dir: str) -> Dict:
    """
    For each step t, log:
      - H_traj for each agent
      - New edges opened (ΔA^(t) > ε)
    Compute mutual information / Pearson r between ΔH and edge formation.
    """
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer = load_base_model(cfg)
    evaluator = MATHEvaluator(cfg, n_samples=50)

    swarm = ECHOSSwarm(model, tokenizer, cfg)

    # Solve a batch of problems to gather statistics
    all_dH: List[float]    = []    # entropy differential before edge opens
    all_formed: List[int]  = []    # 1 if edge formed, 0 if not

    for sample in evaluator.dataset:
        prompt = evaluator.format_prompt(sample)
        _set_seed(cfg.seed)
        swarm_fresh = ECHOSSwarm(model, tokenizer, cfg)
        swarm_fresh.solve(prompt)

        # Extract ΔH values at each step and whether an edge formed
        events = swarm_fresh.topology.edge_formation_events()
        adj_hist = swarm_fresh.topology.adjacency_history()   # (T, N, N)
        H_hist   = swarm_fresh.entropy_tracker.history_tensor()  # (N, T)

        for t in range(1, adj_hist.shape[0]):
            H_prev = H_hist[:, t - 1] if H_hist.shape[1] > t - 1 else H_hist[:, -1]
            for i in range(cfg.swarm.n_agents):
                for j in range(cfg.swarm.n_agents):
                    if i == j:
                        continue
                    dH_ij = float(H_prev[i] - H_prev[j])
                    formed = int(adj_hist[t, i, j] > cfg.swarm.edge_threshold and
                                 adj_hist[t - 1, i, j] <= cfg.swarm.edge_threshold)
                    all_dH.append(dH_ij)
                    all_formed.append(formed)

    # Statistics
    all_dH_arr    = np.array(all_dH)
    all_formed_arr = np.array(all_formed)

    # Pearson correlation
    if all_dH_arr.std() > 1e-8 and all_formed_arr.std() > 1e-8:
        r_val, p_val = pearsonr(all_dH_arr, all_formed_arr)
    else:
        r_val, p_val = 0.0, 1.0

    # Fraction of edge formations preceded by positive ΔH
    pos_dH_mask = all_dH_arr > 0
    formed_given_pos_dH = (
        all_formed_arr[pos_dH_mask].mean()
        if pos_dH_mask.any() else 0.0
    )
    formed_given_neg_dH = (
        all_formed_arr[~pos_dH_mask].mean()
        if (~pos_dH_mask).any() else 0.0
    )

    # Mutual information estimate (binned)
    mi = _mutual_information(all_dH_arr, all_formed_arr, n_bins=20)

    # H4 validation: "entropy spikes precede 85% of edge formations"
    edge_event_dH_signs = [
        1 if dH > 0 else 0
        for dH, f in zip(all_dH, all_formed) if f == 1
    ]
    h4_fraction = float(np.mean(edge_event_dH_signs)) if edge_event_dH_signs else 0.0

    result = {
        "pearson_r":           float(r_val),
        "pearson_p":           float(p_val),
        "mutual_information":  float(mi),
        "formed_given_pos_dH": float(formed_given_pos_dH),
        "formed_given_neg_dH": float(formed_given_neg_dH),
        "h4_edge_preceded_by_high_entropy_fraction": h4_fraction,
        "n_pairs":             len(all_dH),
        "n_edge_events":       int(sum(all_formed)),
    }
    logger.info(
        f"Entropy-Topology: r={r_val:.3f}  MI={mi:.4f}  "
        f"H4={h4_fraction:.3f} (target ≥0.85)"
    )

    path = os.path.join(output_dir, "entropy_topology_correlation.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ═════════════════════════════════════════════════════════════════
# 5.2  Epistemic Filter Efficacy (Adversarial Attack)
# ═════════════════════════════════════════════════════════════════

def run_adversarial_attack(cfg: ECHOSConfig, output_dir: str) -> Dict:
    """
    Initialize 3/15 agents with rogue adapters.
    Measure contamination rate and quarantine success.
    Compare ECHOS (full dual gate) vs cosine-only vs no filter.
    """
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer = load_base_model(cfg)

    n_adversarial    = max(1, int(cfg.swarm.n_agents * cfg.adversarial_fraction))
    adversarial_ids  = list(range(n_adversarial))

    evaluator = AdversarialMathQAEvaluator(
        cfg, n_samples=100,
        adversarial_fraction=cfg.adversarial_fraction
    )

    results = {}

    for condition, gate_settings in [
        ("full_dual_gate",   {"use_dual_gate": True,  "use_epistemic_cutoff": True}),
        ("cosine_only",      {"use_dual_gate": True,  "use_epistemic_cutoff": False}),
        ("no_filter",        {"use_dual_gate": False, "use_epistemic_cutoff": False}),
    ]:
        logger.info(f"\n── Adversarial condition: {condition} ──")
        cond_cfg = copy.deepcopy(cfg)
        for k, v in gate_settings.items():
            setattr(cond_cfg, k, v)

        _set_seed(cfg.seed)
        swarm = ECHOSSwarm(
            model, tokenizer, cond_cfg,
            adversarial_agent_ids=adversarial_ids,
        )
        swarm.set_adversarial_agents(adversarial_ids)

        robustness = evaluator.evaluate_robustness(swarm, adversarial_ids)

        # Also measure: edge weights from adversarial → clean agents over time
        quarantine_scores = _measure_quarantine_success(
            swarm, adversarial_ids, cfg.swarm.edge_threshold
        )

        results[condition] = {
            **robustness,
            "quarantine_scores": quarantine_scores,
        }
        logger.info(
            f"  correct={robustness['correct_rate']:.4f}  "
            f"contamination={robustness['contamination_rate']:.4f}  "
            f"robustness_score={robustness['robustness_score']:.4f}"
        )

    path = os.path.join(output_dir, "adversarial_attack.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Adversarial results → {path}")
    return results


def _measure_quarantine_success(
    swarm: ECHOSSwarm,
    adversarial_ids: List[int],
    edge_threshold: float,
) -> Dict:
    """
    Checks adjacency history for adversarial → clean edges.
    Returns fraction of such edges that were successfully suppressed (≤ threshold).
    """
    adj_hist = swarm.topology.adjacency_history()   # (T, N, N)
    N        = swarm.cfg.swarm.n_agents
    clean_ids = [i for i in range(N) if i not in adversarial_ids]

    suppressed = 0
    total      = 0
    for t in range(adj_hist.shape[0]):
        for adv in adversarial_ids:
            for clean in clean_ids:
                # Edge from adversarial → clean agent (adv as source, clean as dest)
                weight = float(adj_hist[t, clean, adv])
                total += 1
                if weight <= edge_threshold:
                    suppressed += 1

    return {
        "suppression_fraction": suppressed / max(1, total),
        "total_edge_checks":    total,
        "suppressed_count":     suppressed,
    }


# ═════════════════════════════════════════════════════════════════
# 5.3  Tangent Space Verification (TIES vs naive merge)
# ═════════════════════════════════════════════════════════════════

def run_tangent_space_verification(cfg: ECHOSConfig, output_dir: str) -> Dict:
    """
    For pairs of agents, measure:
      - Cosine similarity of ΔW_i and ΔW_j before merging
      - After TIES-merge: cosine similarity of merged ΔW with each original
      - After naive-merge: same metric
    Prediction: TIES maintains higher cosine sim (orthogonal, non-interfering).
    """
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer = load_base_model(cfg)
    swarm = ECHOSSwarm(model, tokenizer, cfg)

    # Solve a few problems to get realistic adapter states
    evaluator = MATHEvaluator(cfg, n_samples=20)
    for sample in evaluator.dataset[:5]:
        swarm.solve(evaluator.format_prompt(sample))

    results = {"pairs": []}

    for layer_name in list(swarm.layer_specs.keys())[:3]:   # first 3 layers
        for i in range(min(5, cfg.swarm.n_agents)):
            for j in range(i + 1, min(5, cfg.swarm.n_agents)):
                d_i = swarm.agents[i].adapters[layer_name].delta()
                d_j = swarm.agents[j].adapters[layer_name].delta()

                pre_cos = _cosine_sim_flat(d_i, d_j)

                # TIES merge
                ties_merged = ties_merge([d_i, d_j], cfg.swarm.trim_fraction)
                ties_cos_i  = _cosine_sim_flat(ties_merged, d_i)
                ties_cos_j  = _cosine_sim_flat(ties_merged, d_j)

                # Naive mean merge
                naive_merged = naive_mean_merge([d_i, d_j])
                naive_cos_i  = _cosine_sim_flat(naive_merged, d_i)
                naive_cos_j  = _cosine_sim_flat(naive_merged, d_j)

                results["pairs"].append({
                    "layer":      layer_name,
                    "agent_i":    i,
                    "agent_j":    j,
                    "pre_cos":    float(pre_cos),
                    "ties_cos_i": float(ties_cos_i),
                    "ties_cos_j": float(ties_cos_j),
                    "naive_cos_i": float(naive_cos_i),
                    "naive_cos_j": float(naive_cos_j),
                })

    # Aggregate
    pairs = results["pairs"]
    ties_mean  = np.mean([max(p["ties_cos_i"],  p["ties_cos_j"])  for p in pairs])
    naive_mean = np.mean([max(p["naive_cos_i"], p["naive_cos_j"]) for p in pairs])
    results["summary"] = {
        "ties_mean_max_cos":  float(ties_mean),
        "naive_mean_max_cos": float(naive_mean),
        "ties_advantage":     float(ties_mean - naive_mean),
    }
    logger.info(
        f"Tangent Space: TIES cos={ties_mean:.4f}  Naive cos={naive_mean:.4f}  "
        f"Advantage={ties_mean - naive_mean:+.4f}"
    )

    path = os.path.join(output_dir, "tangent_space.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return results


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _cosine_sim_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().float()
    bf = b.flatten().float()
    return float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())


def _mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """Estimate mutual information via histogram binning."""
    try:
        from sklearn.metrics import mutual_info_score
        x_binned = np.digitize(x, bins=np.linspace(x.min(), x.max(), n_bins))
        return float(mutual_info_score(x_binned, y))
    except ImportError:
        # Fallback: correlation coefficient as proxy
        if x.std() > 0 and y.std() > 0:
            return float(abs(np.corrcoef(x, y)[0, 1]))
        return 0.0


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
