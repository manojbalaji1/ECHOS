"""
echos/swarm.py
Orchestrates the full ECHOS algorithm (Algorithm 1).

Layout:
  GPU 0  (base_model_device) – Θ_base, forward pass, generation
  GPU 1  (adapter_device)    – A_i, B_i matrices for all N agents,
                               adjacency matrix, entropy tracker
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from config import ECHOSConfig
from echos.agent import (
    ECHOSAgent, LoRAHookManager,
    generate_with_agent, get_layer_specs,
)
from echos.entropy import TrajectoryEntropyTracker
from echos.topology import DynamicTopology
from echos.merging import (
    ties_merge, naive_mean_merge,
    decompose_to_lora, truncated_svd_projection,
)
from echos.flops import FLOPsLog, StepFLOPs, measure_peak_vram, reset_vram_stats

logger = logging.getLogger(__name__)


class ECHOSSwarm:
    """
    Full ECHOS swarm.  Call `solve(prompt)` to run Algorithm 1.
    """

    def __init__(
        self,
        model,
        tokenizer,
        cfg: ECHOSConfig,
        adversarial_agent_ids: Optional[List[int]] = None,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.adversarial_ids = set(adversarial_agent_ids or [])

        hw = cfg.hardware
        lora_cfg = cfg.lora
        swarm_cfg = cfg.swarm

        # ── Discover LoRA target layers ─────────────────────────
        target_sfx = set(lora_cfg.target_modules)
        self.layer_specs = get_layer_specs(model, target_sfx)
        logger.info(f"LoRA targets: {len(self.layer_specs)} linear layers")

        # ── Create N agents ─────────────────────────────────────
        self.agents: List[ECHOSAgent] = [
            ECHOSAgent(
                agent_id       = i,
                layer_specs    = self.layer_specs,
                lora_cfg       = lora_cfg,
                adapter_device = hw.adapter_device,
                compute_dtype  = hw.torch_dtype(),
            )
            for i in range(swarm_cfg.n_agents)
        ]

        # ── Hook manager (shared, one set of hooks at a time) ───
        self.hook_manager = LoRAHookManager(
            model, target_sfx, hw.base_model_device
        )

        # ── Topology & entropy tracker ──────────────────────────
        self.topology = DynamicTopology(
            swarm_cfg.n_agents, swarm_cfg,
            device=hw.adapter_device,
        )
        self.entropy_tracker = TrajectoryEntropyTracker(swarm_cfg.n_agents)

        # ── FLOPs log ──────────────────────────────────────────
        self.flops_log = FLOPsLog()

        # ── Analysis storage ───────────────────────────────────
        self.step_outputs: List[Dict] = []

    # ─────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────

    def solve(self, prompt: str) -> Dict:
        """
        Run ECHOS on a single prompt.  Returns a dict with:
          best_answer, all_answers, step_outputs, flops_log,
          entropy_history, adjacency_history.
        """
        cfg   = self.cfg
        sw    = cfg.swarm
        hw    = cfg.hardware

        enc = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = enc["input_ids"]

        for t in range(sw.max_steps):
            t0 = time.time()
            reset_vram_stats(hw.base_model_device)

            # ══════════════════════════════════════════════════
            # Phase 1: Localised Generation & State Extraction
            # ══════════════════════════════════════════════════
            step_data: Dict = {"step": t, "outputs": {}}

            hidden_states = []   # (N, d_model)
            flat_deltas   = []   # (N, D_flat)

            for i, agent in enumerate(self.agents):
                text, entropy, h = generate_with_agent(
                    agent, prompt_ids, self.model, self.tokenizer,
                    self.hook_manager, cfg,
                )
                agent.trajectory_entropy = entropy
                agent.hidden_state       = h
                agent.last_output        = text

                self.entropy_tracker.update(i, entropy)
                hidden_states.append(h)
                flat_deltas.append(agent.flat_delta_concat())

                step_data["outputs"][i] = {"text": text, "entropy": entropy}
                logger.debug(f"  Agent {i:02d}: H={entropy:.4f}  {text[:80]}…")

            H_tensor  = self.entropy_tracker.as_tensor(hw.adapter_device)
            hs_matrix = torch.stack(hidden_states, dim=0)   # (N, d)
            dW_matrix = torch.stack(flat_deltas,   dim=0)   # (N, D)

            # ══════════════════════════════════════════════════
            # Phase 2: Topology Adaptation & Epistemic Gating
            # ══════════════════════════════════════════════════
            phi = self.topology.epistemic_filter(
                h              = hs_matrix,
                delta_W        = dW_matrix,
                use_dual_gate  = cfg.use_dual_gate,
                use_hard_cutoff = cfg.use_epistemic_cutoff,
            )

            self.topology.update(
                entropy            = H_tensor,
                phi                = phi,
                use_entropy_routing = cfg.use_entropy_routing,
            )

            step_data["adjacency"] = self.topology.A.cpu()
            step_data["entropies"] = H_tensor.cpu().tolist()
            step_data["phi"]       = phi.cpu()

            # ══════════════════════════════════════════════════
            # Phase 3: TIES-Gossip & Rank-Preserving Projection
            # ══════════════════════════════════════════════════
            for i, agent in enumerate(self.agents):
                peers = self.topology.active_peers(i)
                if not peers:
                    continue

                # Collect peer deltas per layer
                for layer_name, my_adapter in agent.adapters.items():
                    peer_deltas  = []
                    peer_weights = []

                    for j, w in peers:
                        peer_delta = self.agents[j].adapters[layer_name].delta()
                        peer_deltas.append(peer_delta.to(hw.adapter_device))
                        peer_weights.append(w)

                    if not peer_deltas:
                        continue

                    # Merge
                    if cfg.use_ties_merge:
                        merged_delta = ties_merge(
                            peer_deltas, sw.trim_fraction, peer_weights
                        )
                    else:
                        merged_delta = naive_mean_merge(peer_deltas, peer_weights)

                    # Update dense weight: W_dense = A_i B_i + η * merged_delta
                    W_dense = my_adapter.delta() + sw.merge_rate * merged_delta

                    # Project back to rank r
                    if cfg.use_rsvd:
                        A_new, B_new = decompose_to_lora(
                            W_dense, my_adapter.rank, cfg.swarm_cfg_rsvd_oversampling
                        )
                    else:
                        A_new, B_new = truncated_svd_projection(W_dense, my_adapter.rank)

                    my_adapter.set_matrices(A_new, B_new)

                self.topology.zero_edges_after_merge(i)

            # ── Record step ────────────────────────────────────
            wall = time.time() - t0
            peak_vram = measure_peak_vram(hw.base_model_device)
            self.flops_log.append(StepFLOPs(
                step=t, traj_length=sw.traj_window,
                n_agents=sw.n_agents, method="echos",
                wall_time_s=wall, peak_vram_gb=peak_vram,
            ))
            self.step_outputs.append(step_data)

            # ── Convergence check ──────────────────────────────
            if self._check_consensus(step_data["outputs"]):
                logger.info(f"Consensus reached at step {t}")
                break

        # Return best answer (lowest terminal entropy)
        best_idx = int(H_tensor.argmin())
        return {
            "best_answer":       self.agents[best_idx].last_output,
            "best_agent":        best_idx,
            "all_answers":       {i: a.last_output for i, a in enumerate(self.agents)},
            "step_outputs":      self.step_outputs,
            "flops_log":         self.flops_log,
            "entropy_history":   self.entropy_tracker.history_tensor().cpu(),
            "adjacency_history": self.topology.adjacency_history().cpu(),
            "edge_events":       self.topology.edge_formation_events(),
            "n_steps":           len(self.step_outputs),
        }

    def _check_consensus(self, outputs: Dict) -> bool:
        """Simple string-based majority vote for early stopping."""
        answers = [v["text"].strip().lower() for v in outputs.values()]
        if not answers:
            return False
        most_common = max(set(answers), key=answers.count)
        fraction    = answers.count(most_common) / len(answers)
        return fraction >= 0.8   # 80% agreement → consensus

    @property
    def swarm_cfg_rsvd_oversampling(self):
        return self.cfg.swarm.rsvd_oversampling

    # Expose for ablation experiments
    def set_adversarial_agents(
        self,
        agent_ids: List[int],
        wrong_lora_scales: float = 5.0,
    ) -> None:
        """
        Simulates adversarial agents by scaling their LoRA B matrices
        to produce confidently wrong (out-of-distribution) hidden states.
        """
        for idx in agent_ids:
            for adapter in self.agents[idx].adapters.values():
                # Inject random high-magnitude B to simulate rogue adapter
                adapter.B = torch.randn_like(adapter.B) * wrong_lora_scales
            logger.info(f"Agent {idx} set to adversarial mode")
