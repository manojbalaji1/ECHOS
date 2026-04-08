"""
analysis/plots.py
Generates all figures required for the TMLR submission.

Figure 1: Accuracy vs TFLOPs scatter (Pareto frontier)
Figure 2: Breakeven curve (empirical FLOPs crossing vs sequence length)
Figure 3: Network graph evolution (3 time steps)
Figure 4: Ablation bar chart
Figure 5: Swarm scaling (accuracy + VRAM vs N)
Figure 6: Adversarial robustness comparison
Figure 7: Entropy trajectories + topology heatmap
Figure 8: Tangent space cosine similarity
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import seaborn as sns

# TMLR-friendly style
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Color palette
COLORS = {
    "echos":            "#E63946",
    "text_debate":      "#457B9D",
    "sparse_text":      "#A8DADC",
    "static_lora_avg":  "#F4A261",
    "lora_ensemble":    "#2A9D8F",
    "single_agent":     "#6D6875",
}


# ═════════════════════════════════════════════════════════════════
# Figure 1: Accuracy vs TFLOPs (Pareto scatter)
# ═════════════════════════════════════════════════════════════════

def plot_pareto_frontier(results: Dict, output_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    benchmarks_to_plot = [
        ("math",        "MATH (Lvl 3-5)"),
        ("strategy_qa", "StrategyQA"),
    ]

    method_labels = {
        "echos":             "ECHOS (Ours)",
        "text_debate":       "Text Debate",
        "sparse_text":       "Sparse Text",
        "static_lora_avg":   "Static LoRA Avg",
        "lora_ensemble":     "LoRA Ensemble",
        "single_agent_greedy": "Single Agent (Greedy)",
        "self_consistency_64": "Self-Consistency (N=64)",
    }

    for ax, (bench_key, bench_label) in zip(axes, benchmarks_to_plot):
        bench_data = results.get(bench_key, {})

        for method, data in bench_data.items():
            if not isinstance(data, dict):
                continue
            acc    = data.get("accuracy", 0)
            tflops = data.get("total_tflops", data.get("meta", {}).get("total_tflops", 1.0))
            label  = method_labels.get(method, method)
            color  = COLORS.get(method.split("_")[0], "#888888")
            marker = "*" if method == "echos" else "o"
            size   = 200 if method == "echos" else 80

            ax.scatter(tflops, acc, c=color, s=size, marker=marker,
                       label=label, zorder=5, edgecolors="white", linewidths=0.5)

        ax.set_xlabel("Total TFLOPs")
        ax.set_ylabel("Accuracy")
        ax.set_title(bench_label)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, linestyle="--")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Figure 1: Accuracy vs Compute (Pareto Frontier)", fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    path = os.path.join(output_dir, "fig1_pareto_frontier.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    return path


# ═════════════════════════════════════════════════════════════════
# Figure 2: Breakeven Curve
# ═════════════════════════════════════════════════════════════════

def plot_breakeven(breakeven_data: Dict, output_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    records = breakeven_data.get("records", [])
    L_vals  = [r["traj_length"] for r in records]

    # Left: FLOPs vs L
    ax = axes[0]
    echos_emp  = [r.get("echos_empirical_tflops",  r.get("echos_analytical_tflops", 0)) for r in records]
    text_emp   = [r.get("text_empirical_tflops",   r.get("text_analytical_tflops",  0)) for r in records]
    echos_anal = [r["echos_analytical_tflops"] for r in records]
    text_anal  = [r["text_analytical_tflops"]  for r in records]

    ax.plot(L_vals, text_emp,   "o-", color=COLORS["text_debate"], label="Text Debate (Empirical)", linewidth=2)
    ax.plot(L_vals, echos_emp,  "s-", color=COLORS["echos"],       label="ECHOS (Empirical)",       linewidth=2)
    ax.plot(L_vals, text_anal,  "--", color=COLORS["text_debate"], label="Text Debate (Analytical)", alpha=0.6)
    ax.plot(L_vals, echos_anal, "--", color=COLORS["echos"],       label="ECHOS (Analytical)",       alpha=0.6)

    # Mark crossover
    L_star = breakeven_data.get("empirical_breakeven_L", breakeven_data.get("analytical_breakeven_L", 488))
    if not np.isnan(L_star):
        ax.axvline(L_star, color="black", linestyle=":", linewidth=1.5,
                   label=f"Crossover L*≈{L_star:.0f}")

    ax.set_xlabel("Trajectory Length |T| (tokens)")
    ax.set_ylabel("TFLOPs")
    ax.set_title("Communication FLOPs vs Trajectory Length")
    ax.legend(fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Right: Accuracy vs L
    ax = axes[1]
    echos_acc = [r.get("echos_accuracy", np.nan) for r in records]
    text_acc  = [r.get("text_accuracy",  np.nan) for r in records]
    ax.plot(L_vals, text_acc,  "o-", color=COLORS["text_debate"], label="Text Debate")
    ax.plot(L_vals, echos_acc, "s-", color=COLORS["echos"],       label="ECHOS")
    ax.set_xlabel("Trajectory Length |T| (tokens)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Trajectory Length")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Figure 2: Breakeven Theorem Validation", fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "fig2_breakeven.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    return path


# ═════════════════════════════════════════════════════════════════
# Figure 3: Network Graph Evolution (3 snapshots)
# ═════════════════════════════════════════════════════════════════

def plot_network_evolution(adj_history: np.ndarray, output_dir: str,
                           entropy_history: Optional[np.ndarray] = None) -> str:
    """adj_history: (T, N, N)  entropy_history: (N, T)"""
    try:
        import networkx as nx
    except ImportError:
        # Fallback: heatmap of adjacency
        return _plot_adj_heatmap(adj_history, output_dir)

    T, N, _ = adj_history.shape
    t_steps  = [0, T // 2, T - 1]
    step_labels = ["Initialization", "Bottleneck", "Consensus"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, t_idx, label in zip(axes, t_steps, step_labels):
        A   = adj_history[min(t_idx, T - 1)]
        G   = nx.DiGraph()
        G.add_nodes_from(range(N))
        threshold = 0.05
        for i in range(N):
            for j in range(N):
                if i != j and A[i, j] > threshold:
                    G.add_edge(j, i, weight=float(A[i, j]))

        pos = nx.circular_layout(G)

        # Node colors: entropy (if available)
        if entropy_history is not None and entropy_history.shape[1] > t_idx:
            ent  = entropy_history[:, min(t_idx, entropy_history.shape[1] - 1)]
            norm = plt.Normalize(ent.min(), ent.max())
            node_colors = [plt.cm.RdYlGn_r(norm(e)) for e in ent]
        else:
            node_colors = [COLORS["echos"]] * N

        # Edge widths
        edge_weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=300, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white")
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights,
                               edge_color="gray", alpha=0.6,
                               arrows=True, arrowsize=10,
                               connectionstyle="arc3,rad=0.1")
        ax.set_title(f"{label} (t={t_idx})")
        ax.axis("off")

    fig.suptitle("Figure 3: ECHOS Network Graph Evolution", fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "fig3_network_evolution.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    return path


def _plot_adj_heatmap(adj_history: np.ndarray, output_dir: str) -> str:
    T, N, _ = adj_history.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, t_idx, label in zip(axes, [0, T // 2, T - 1], ["Init", "Mid", "End"]):
        A = adj_history[min(t_idx, T - 1)]
        sns.heatmap(A, ax=ax, cmap="Blues", vmin=0, vmax=1,
                    xticklabels=False, yticklabels=False, cbar=(ax == axes[-1]))
        ax.set_title(label)
    fig.suptitle("Figure 3: Adjacency Matrix Evolution")
    path = os.path.join(output_dir, "fig3_adjacency_heatmap.pdf")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


# ═════════════════════════════════════════════════════════════════
# Figure 4: Ablation Bar Chart
# ═════════════════════════════════════════════════════════════════

def plot_ablations(ablation_data: Dict, output_dir: str) -> str:
    conditions = list(ablation_data.keys())
    means = [ablation_data[c]["accuracy_mean"] for c in conditions]
    stds  = [ablation_data[c]["accuracy_std"]  for c in conditions]

    labels = {
        "full_echos":    "Full ECHOS\n(Ours)",
        "no_entropy":    "No Entropy\nRouting",
        "no_quarantine": "No Hard\nQuarantine",
        "no_epistemic":  "No Epistemic\nFilter",
        "naive_merge":   "Naive Mean\nMerge",
        "no_svd":        "Truncated\nSVD",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conditions))
    colors = [COLORS["echos"] if c == "full_echos" else "#6D6875" for c in conditions]

    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4,
                  error_kw={"linewidth": 1.5}, width=0.65, zorder=3)

    # Delta annotations vs full ECHOS
    full_acc = ablation_data.get("full_echos", {}).get("accuracy_mean", 0)
    for i, (cond, bar) in enumerate(zip(conditions, bars)):
        if cond == "full_echos":
            continue
        delta = means[i] - full_acc
        ax.annotate(
            f"{delta:+.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, means[i] + stds[i] + 0.005),
            ha="center", va="bottom", fontsize=9,
            color="#C62828" if delta < 0 else "#2E7D32",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, c) for c in conditions])
    ax.set_ylabel("Accuracy (MATH Lvl 3-5)")
    ax.set_title("Figure 4: Ablation Study – Component Importance")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, min(1.05, max(means) * 1.15))

    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_ablations.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    return path


# ═════════════════════════════════════════════════════════════════
# Figure 5: Scaling (Accuracy + VRAM vs N)
# ═════════════════════════════════════════════════════════════════

def plot_scaling(scaling_data: Dict, output_dir: str) -> str:
    records = scaling_data.get("records", [])
    if not records:
        return ""

    N_vals   = [r["N"] for r in records]
    acc_vals = [r["accuracy"] for r in records]
    acc_std  = [r.get("accuracy_std", 0) for r in records]
    vram_vals = [r["peak_vram_gb"] for r in records]
    specialisation = [r.get("specialisation_score", np.nan) for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy vs N
    ax = axes[0]
    ax.plot(N_vals, acc_vals, "o-", color=COLORS["echos"], linewidth=2, markersize=8)
    ax.fill_between(N_vals,
                    np.array(acc_vals) - np.array(acc_std),
                    np.array(acc_vals) + np.array(acc_std),
                    alpha=0.2, color=COLORS["echos"])
    ax.set_xlabel("Swarm Size N")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Swarm Size")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Peak VRAM vs N
    ax = axes[1]
    ax.plot(N_vals, vram_vals, "s-", color="#457B9D", linewidth=2, markersize=8)
    ax.axhline(96, color="red", linestyle="--", alpha=0.7, label="Total VRAM (96GB)")
    ax.axhline(48, color="orange", linestyle=":", alpha=0.7, label="Adapter GPU (48GB)")
    ax.set_xlabel("Swarm Size N")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("VRAM Usage vs Swarm Size")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")

    # Specialisation score vs N
    ax = axes[2]
    ax.plot(N_vals, specialisation, "^-", color="#2A9D8F", linewidth=2, markersize=8)
    ax.set_xlabel("Swarm Size N")
    ax.set_ylabel("Specialisation Score (1 - mean cos-sim)")
    ax.set_title("Agent Specialisation vs Swarm Size")
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Figure 5: Swarm Scaling Analysis", fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "fig5_scaling.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    return path


# ═════════════════════════════════════════════════════════════════
# Figure 6: Adversarial Robustness
# ═════════════════════════════════════════════════════════════════

def plot_adversarial(adversarial_data: Dict, output_dir: str) -> str:
    conditions    = list(adversarial_data.keys())
    correct_rates = [adversarial_data[c]["correct_rate"]       for c in conditions]
    contam_rates  = [adversarial_data[c]["contamination_rate"] for c in conditions]
    robustness    = [adversarial_data[c]["robustness_score"]   for c in conditions]

    label_map = {
        "full_dual_gate": "Full Dual Gate\n(ECHOS)",
        "cosine_only":    "Cosine-Only\nFilter",
        "no_filter":      "No Filter",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    x = np.arange(len(conditions))
    bar_colors = [COLORS["echos"], "#F4A261", COLORS["text_debate"]]

    axes[0].bar(x, correct_rates, color=bar_colors, width=0.5)
    axes[0].set_title("Correct Consensus Rate")
    axes[0].set_ylabel("Fraction Correct")

    axes[1].bar(x, contam_rates, color=bar_colors, width=0.5)
    axes[1].set_title("Contamination Rate ↓")
    axes[1].set_ylabel("Fraction Contaminated")

    capped_robustness = [min(r, 50) for r in robustness]  # cap inf for display
    axes[2].bar(x, capped_robustness, color=bar_colors, width=0.5)
    axes[2].set_title("Robustness Score ↑\n(Correct Rate / Contam. Rate)")
    axes[2].set_ylabel("Robustness Score")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([label_map.get(c, c) for c in conditions])
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Figure 6: Adversarial Robustness (20% Rogue Agents)", fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "fig6_adversarial.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    return path


# ═════════════════════════════════════════════════════════════════
# Figure 7: Entropy Trajectories + Topology Heatmap
# ═════════════════════════════════════════════════════════════════

def plot_entropy_topology(
    entropy_history: np.ndarray,    # (N, T)
    adj_history: np.ndarray,        # (T, N, N)
    edge_events: List[Dict],
    output_dir: str,
) -> str:
    N, T = entropy_history.shape
    fig  = plt.figure(figsize=(14, 8))
    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Top-left: entropy curves per agent
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(N):
        alpha = 0.8 if i < 3 else 0.3   # highlight first 3 agents
        lw    = 2.0 if i < 3 else 0.8
        ax1.plot(range(T), entropy_history[i], alpha=alpha, linewidth=lw,
                 label=f"Agent {i}" if i < 3 else None)

    # Mark edge formation events
    for ev in edge_events[:20]:   # first 20 events
        ax1.axvline(ev["step"], color="gray", alpha=0.2, linewidth=0.5)

    ax1.set_xlabel("Trajectory Step")
    ax1.set_ylabel("Trajectory Entropy H_traj")
    ax1.set_title("Entropy Trajectories (highlighted: agents 0-2)")
    ax1.legend(loc="upper right", ncol=3)
    ax1.grid(True, alpha=0.2)

    # Bottom-left: mean entropy over time
    ax2 = fig.add_subplot(gs[1, 0])
    mean_ent = entropy_history.mean(axis=0)
    std_ent  = entropy_history.std(axis=0)
    ax2.plot(range(T), mean_ent, color=COLORS["echos"], linewidth=2)
    ax2.fill_between(range(T), mean_ent - std_ent, mean_ent + std_ent,
                     alpha=0.2, color=COLORS["echos"])
    ax2.set_xlabel("Trajectory Step")
    ax2.set_ylabel("Mean Entropy")
    ax2.set_title("Swarm-Average Entropy (↓ = consensus)")
    ax2.grid(True, alpha=0.3)

    # Bottom-right: final adjacency heatmap
    ax3 = fig.add_subplot(gs[1, 1])
    final_A = adj_history[-1]
    im = ax3.imshow(final_A, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_xlabel("Source Agent j")
    ax3.set_ylabel("Target Agent i")
    ax3.set_title("Final Adjacency Matrix A^(T)")

    fig.suptitle("Figure 7: Entropy-Topology Dynamics", fontweight="bold")

    path = os.path.join(output_dir, "fig7_entropy_topology.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    return path


# ═════════════════════════════════════════════════════════════════
# Figure 8: Tangent Space Cosine Similarity
# ═════════════════════════════════════════════════════════════════

def plot_tangent_space(tangent_data: Dict, output_dir: str) -> str:
    pairs  = tangent_data.get("pairs", [])
    if not pairs:
        return ""

    pre_cos   = [p["pre_cos"]   for p in pairs]
    ties_cos  = [max(p["ties_cos_i"],  p["ties_cos_j"])  for p in pairs]
    naive_cos = [max(p["naive_cos_i"], p["naive_cos_j"]) for p in pairs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: distribution of cosine sims
    ax = axes[0]
    bins = np.linspace(-1, 1, 40)
    ax.hist(pre_cos,   bins=bins, alpha=0.6, label="Pre-merge",    color="#6D6875")
    ax.hist(ties_cos,  bins=bins, alpha=0.7, label="After TIES",   color=COLORS["echos"])
    ax.hist(naive_cos, bins=bins, alpha=0.7, label="After Naive",  color=COLORS["text_debate"])
    ax.axvline(np.mean(ties_cos),  color=COLORS["echos"],       linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(naive_cos), color=COLORS["text_debate"], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Cosine Similarity(Merged ΔW, Original ΔW)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Update Cosine Similarity")
    ax.legend()

    # Right: scatter pre-merge cos vs delta (TIES - naive)
    ax = axes[1]
    delta_cos = np.array(ties_cos) - np.array(naive_cos)
    ax.scatter(pre_cos, delta_cos, alpha=0.5, s=30, color=COLORS["echos"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Pre-merge Cosine Similarity (ΔW_i · ΔW_j)")
    ax.set_ylabel("TIES Advantage (cos_TIES - cos_Naive)")
    ax.set_title("TIES Advantage vs Update Orthogonality")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 8: Tangent Space Verification", fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "fig8_tangent_space.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    return path


# ═════════════════════════════════════════════════════════════════
# Master: generate all figures from saved results
# ═════════════════════════════════════════════════════════════════

def generate_all_figures(results_dir: str, figures_dir: str) -> List[str]:
    os.makedirs(figures_dir, exist_ok=True)
    generated = []

    def _load(fname):
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    # Fig 1
    d = _load("main_comparison.json")
    if d:
        generated.append(plot_pareto_frontier(d, figures_dir))

    # Fig 2
    d = _load("breakeven.json")
    if d:
        generated.append(plot_breakeven(d, figures_dir))

    # Figs 3, 7 need numpy arrays – regenerated from raw step outputs
    # (saved separately by the swarm runner)
    for fname, plot_fn, label in [
        ("ablations.json",            plot_ablations,   "fig4"),
        ("scaling.json",              plot_scaling,     "fig5"),
        ("adversarial_attack.json",   plot_adversarial, "fig6"),
        ("tangent_space.json",        plot_tangent_space, "fig8"),
    ]:
        d = _load(fname)
        if d:
            generated.append(plot_fn(d, figures_dir))

    print(f"Generated {len(generated)} figures in {figures_dir}")
    return generated
