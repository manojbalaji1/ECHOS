# ECHOS: Experiment Suite for TMLR Submission

**Entropy-Driven Communication and Heuristic Organization in Swarms via Ephemeral LoRA Gossip**

This repository contains the full experimental suite to validate the ECHOS paper for TMLR submission.
Every table, figure, and hypothesis test in the paper is produced by this code.

---

## Hardware Requirements

| Component | Spec |
|-----------|------|
| GPU 0 (base model) | L40 48 GB — hosts Θ_base, runs forward passes |
| GPU 1 (adapters)   | L40 48 GB — hosts all N LoRA states, adjacency matrix |
| Total VRAM | 96 GB |
| Estimated GPU-hours | ~40 hrs for full suite |

---

## Installation

```bash
# 1. Clone / unzip the repo
cd echos_experiments

# 2. Create environment
conda create -n echos python=3.11 -y
conda activate echos

# 3. Install PyTorch for your CUDA version (12.x example)
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining deps
pip install -r requirements.txt

# 5. (Optional) SWE-bench harness — requires Docker
pip install swebench
```

### Model Access

Set your HuggingFace token for gated models (Llama):
```bash
export HF_TOKEN=<your_token>
huggingface-cli login
```

---

## Quick Start

### Smoke test (verifies the whole pipeline in ~5 min)
```bash
python run_experiments.py --smoke
```

### Full suite (default: Qwen2.5-7B, bf16)
```bash
python run_experiments.py --all
```

### Change base model or quantization
```bash
# Llama-3.1-8B in 4-bit NF4 (fits in ~20 GB)
python run_experiments.py --all \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --quant fp4

# Mistral-7B in bf16
python run_experiments.py --all \
    --model "mistralai/Mistral-7B-Instruct-v0.3" \
    --quant bf16

# DeepSeek-R1 distil
python run_experiments.py --all \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --quant bf16
```

### Run specific experiments
```bash
python run_experiments.py --exp ablations
python run_experiments.py --exp breakeven --n_samples 50
python run_experiments.py --exp main_comparison ablations plots stats
```

---

## File Structure

```
echos_experiments/
│
├── config.py                   # All hyperparameters and experiment flags
├── run_experiments.py          # Master CLI entry point
├── requirements.txt
│
├── echos/                      # Core ECHOS algorithm
│   ├── model_loader.py         # Load base model (fp4/int8/bf16/fp16/fp32)
│   ├── agent.py                # Per-agent LoRA state + forward hooks
│   ├── entropy.py              # Trajectory entropy H_traj
│   ├── topology.py             # Dynamic graph G(t), epistemic filter Φ
│   ├── merging.py              # TIES-merge + Randomized SVD
│   ├── swarm.py                # Algorithm 1 orchestrator
│   └── flops.py                # Analytical + empirical FLOPs measurement
│
├── baselines/
│   └── all_baselines.py        # All 5 baselines (CoT, TextDebate, Sparse,
│                               #   StaticLoRA, LoRAEnsemble)
│
├── benchmarks/
│   ├── base_eval.py            # Abstract evaluator + answer extractors
│   ├── math_eval.py            # MATH (levels 3-5)
│   ├── gpqa_eval.py            # GPQA Diamond
│   ├── strategy_qa.py          # StrategyQA + AdversarialMathQA
│   └── swe_bench_eval.py       # SWE-bench Lite
│
├── experiments/
│   ├── main_comparison.py      # Table 1: accuracy + FLOPs across methods
│   ├── ablations.py            # Table 2: component ablations
│   ├── breakeven.py            # Figure 2: breakeven theorem validation
│   ├── scaling.py              # Figure 5: swarm scaling (N sweep)
│   ├── mechanistic.py          # Figures 7, 8: entropy-topology, tangent space
│   └── hp_sweep.py             # HP grid search (γ, τ, k)
│
└── analysis/
    ├── plots.py                # All 8 paper figures (matplotlib, TMLR style)
    └── stats.py                # t-tests, Cohen's d, bootstrap CIs, LaTeX tables
```

---

## Configuration

All hyperparameters live in `config.py`. Key knobs:

```python
# Base model (any HF model ID)
model_name = "Qwen/Qwen2.5-7B-Instruct"

# Quantization
hardware.quant_mode = "bf16"   # fp4 | int8 | bf16 | fp16 | fp32

# Swarm
swarm.n_agents      = 15       # N
swarm.traj_window   = 32       # |T| tokens per step
swarm.temperature   = 0.5      # τ (entropy routing)
swarm.cosine_threshold = 0.8   # γ (epistemic filter)
swarm.trim_fraction = 0.3      # k (TIES-merge top-k%)

# LoRA
lora.r              = 8        # rank
lora.target_modules = ["q_proj", "v_proj", "down_proj", "up_proj"]

# Ablation switches
use_entropy_routing = True
use_dual_gate       = True
use_epistemic_cutoff = True    # hard cutoff in Φ
use_ties_merge      = True     # False → naive mean
use_rsvd            = True     # False → truncated SVD
```

---

## Paper → Code Mapping

| Paper Section | Code |
|--------------|------|
| §2.1 Swarm Init | `echos/agent.py :: ECHOSAgent.__init__` |
| §2.2 Trajectory Entropy | `echos/entropy.py :: entropy_from_logits` |
| §2.3 Dual-Gated Filter Φ | `echos/topology.py :: DynamicTopology.epistemic_filter` |
| §2.4 TIES-Gossip + rSVD | `echos/merging.py :: ties_merge, decompose_to_lora` |
| §2.5 Tangent Space | `experiments/mechanistic.py :: run_tangent_space_verification` |
| Algorithm 1 | `echos/swarm.py :: ECHOSSwarm.solve` |
| §3 FLOPs Analysis | `echos/flops.py :: text_debate_flops, echos_gossip_flops` |
| §3.3 Breakeven Theorem | `experiments/breakeven.py` |
| §4.1 Main Comparison | `experiments/main_comparison.py` |
| §4.2 Ablations | `experiments/ablations.py` |
| §4.3 Breakeven Validation | `experiments/breakeven.py` |
| §4.4 Scaling | `experiments/scaling.py` |
| §5.1 Entropy-Topology | `experiments/mechanistic.py :: run_entropy_topology_correlation` |
| §5.2 Adversarial Attack | `experiments/mechanistic.py :: run_adversarial_attack` |
| §5.3 Tangent Space | `experiments/mechanistic.py :: run_tangent_space_verification` |
| Table 1 LaTeX | `analysis/stats.py :: generate_main_table_latex` |
| Table 2 LaTeX | `analysis/stats.py :: generate_ablation_table_latex` |
| Figures 1-8 | `analysis/plots.py` |

---

## Reproducibility

- All stochastic components seeded via `--seed` (default: 0)
- Full results use 5 seeds: `{0, 1, 2, 3, 4}` (ablations use 3)
- rSVD randomness is seeded per the global seed
- All FLOPs measured via `torch.utils.flop_counter` (forward only)
- Expected results saved to `results/` (JSON) and `results/figures/` (PDF+PNG)

---

## Expected Outputs

After a full run, `results/` will contain:

```
results/
├── config.json
├── main_comparison.json     # Table 1 data
├── flops_table.json         # Analytical FLOPs breakdown
├── ablations.json           # Table 2 data
├── breakeven.json           # Figure 2 data
├── scaling.json             # Figure 5 data
├── entropy_topology_correlation.json
├── adversarial_attack.json
├── tangent_space.json
├── hp_sweep.json
├── figures/
│   ├── fig1_pareto_frontier.{pdf,png}
│   ├── fig2_breakeven.{pdf,png}
│   ├── fig3_network_evolution.{pdf,png}
│   ├── fig4_ablations.{pdf,png}
│   ├── fig5_scaling.{pdf,png}
│   ├── fig6_adversarial.{pdf,png}
│   ├── fig7_entropy_topology.{pdf,png}
│   └── fig8_tangent_space.{pdf,png}
└── stats/
    ├── stat_tests.json          # t-tests + Cohen's d per benchmark
    ├── hypotheses.json          # H1-H4 validation results
    ├── table1_main.tex          # Ready for copy-paste into paper
    └── table2_ablation.tex
```
