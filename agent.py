"""
echos/agent.py
Represents a single ECHOS agent:  base model Θ_base + LoRA adapter (A_i, B_i).

The base model is shared (read-only, on GPU 0).
The adapter matrices live on the adapter_device (GPU 1).

Generation is done by temporarily materialising the effective weight via:
    W_eff = Θ_base + A_i @ B_i
using forward hooks that add the low-rank perturbation in-place during the
forward pass – avoiding copying the full base model per agent.
"""

from __future__ import annotations
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig as HFGenerationConfig

from config import ECHOSConfig, LoRAConfig
from echos.entropy import entropy_from_logits

logger = logging.getLogger(__name__)

# Names of linear sub-modules we attach adapters to (matched by suffix)
_DEFAULT_TARGET_SUFFIXES = {"q_proj", "v_proj", "down_proj", "up_proj"}


# ─────────────────────────────────────────────────────────────────
# LoRA adapter for one weight matrix
# ─────────────────────────────────────────────────────────────────

class LoRAAdapter:
    """
    Stores A (d_in, r) and B (r, d_out) on adapter_device.
    Can compute the dense delta: ΔW = A @ B.
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        rank: int,
        device: str,
        dtype: torch.dtype,
        lora_alpha: int = 16,
    ):
        self.rank      = rank
        self.d_in      = d_in
        self.d_out     = d_out
        self.device    = device
        self.dtype     = dtype
        self.scale     = lora_alpha / rank

        # Kaiming init for A, zero init for B (standard LoRA)
        self.A = torch.empty(d_in, rank, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.zeros(rank, d_out, device=device, dtype=dtype)

    def delta(self) -> torch.Tensor:
        """Dense ΔW = A @ B  (d_in, d_out) on adapter_device."""
        return (self.A @ self.B) * self.scale

    def flat_delta(self) -> torch.Tensor:
        """Flattened ΔW for distance / similarity computations."""
        return self.delta().flatten()

    def set_matrices(self, A: torch.Tensor, B: torch.Tensor) -> None:
        self.A = A.to(self.device, dtype=self.dtype)
        self.B = B.to(self.device, dtype=self.dtype)


# ─────────────────────────────────────────────────────────────────
# Per-agent state
# ─────────────────────────────────────────────────────────────────

class ECHOSAgent:
    """
    One agent = a dict of LoRAAdapters (one per targeted linear layer)
    + the latest trajectory entropy + the latest final hidden state.
    """

    def __init__(
        self,
        agent_id: int,
        layer_specs: Dict[str, Tuple[int, int]],  # {layer_name: (d_in, d_out)}
        lora_cfg: LoRAConfig,
        adapter_device: str,
        compute_dtype: torch.dtype,
    ):
        self.id            = agent_id
        self.adapter_device = adapter_device
        self.compute_dtype  = compute_dtype

        # One LoRA adapter per targeted layer
        self.adapters: Dict[str, LoRAAdapter] = {
            name: LoRAAdapter(
                d_in=d_in, d_out=d_out,
                rank=lora_cfg.r,
                lora_alpha=lora_cfg.lora_alpha,
                device=adapter_device,
                dtype=compute_dtype,
            )
            for name, (d_in, d_out) in layer_specs.items()
        }

        # Updated after each trajectory step
        self.trajectory_entropy: float = float("inf")
        self.hidden_state: Optional[torch.Tensor] = None    # (d_model,)
        self.last_output: Optional[str] = None

    def flat_delta_concat(self) -> torch.Tensor:
        """Concatenated flattened deltas across all layers → used for topology."""
        return torch.cat([a.flat_delta() for a in self.adapters.values()])

    def per_layer_deltas(self) -> Dict[str, torch.Tensor]:
        """Dict of {name: (d_in, d_out) dense ΔW} for TIES-merge."""
        return {name: a.delta() for name, a in self.adapters.items()}


# ─────────────────────────────────────────────────────────────────
# Forward hooks for LoRA injection
# ─────────────────────────────────────────────────────────────────

class LoRAHookManager:
    """
    Temporarily installs forward hooks on the base model's linear layers
    to add the agent's adapter delta during the forward pass.
    No weight copying needed.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        target_suffixes: set,
        base_device: str,
    ):
        self.model          = model
        self.target_suffixes = target_suffixes
        self.base_device    = base_device
        self._hooks: List   = []
        self._current_adapters: Dict[str, LoRAAdapter] = {}

        # Pre-build map: full module name → module for targeted layers
        self.name_to_module: Dict[str, nn.Module] = {
            name: module
            for name, module in model.named_modules()
            if any(name.endswith(suf) for suf in target_suffixes)
            and isinstance(module, nn.Linear)
        }

    def _make_hook(self, layer_name: str):
        def hook(module, input, output):
            if layer_name not in self._current_adapters:
                return output
            adapter = self._current_adapters[layer_name]
            # Move delta to base_device for the addition
            delta = adapter.delta().to(self.base_device, dtype=output.dtype)
            # output shape: (batch, seq, d_out) or (batch, d_out)
            # input[0] shape: (batch, seq, d_in) or (batch, d_in)
            x = input[0]
            lora_out = x @ delta                    # broadcasts correctly
            return output + lora_out
        return hook

    def install(self, agent: ECHOSAgent) -> None:
        """Register hooks for this agent's adapters."""
        assert not self._hooks, "Previous hooks not removed – call remove() first."
        self._current_adapters = agent.adapters
        for name, module in self.name_to_module.items():
            h = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._current_adapters = {}


# ─────────────────────────────────────────────────────────────────
# Generation with per-agent adapter
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_with_agent(
    agent: ECHOSAgent,
    prompt_ids: torch.Tensor,           # (1, L_prompt)
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    hook_manager: LoRAHookManager,
    cfg: ECHOSConfig,
) -> Tuple[str, float, torch.Tensor]:
    """
    Generates a trajectory for `agent` and returns:
      (text, trajectory_entropy, final_hidden_state)
    """
    base_device = cfg.hardware.base_model_device
    gen_cfg     = cfg.generation

    prompt_ids = prompt_ids.to(base_device)

    # Install agent's adapter hooks
    hook_manager.install(agent)
    try:
        out = model.generate(
            input_ids        = prompt_ids,
            max_new_tokens   = gen_cfg.max_new_tokens,
            do_sample        = gen_cfg.do_sample,
            temperature      = gen_cfg.temperature,
            top_p            = gen_cfg.top_p,
            repetition_penalty = gen_cfg.repetition_penalty,
            output_scores    = True,           # needed for entropy
            output_hidden_states = True,        # needed for Φ filter
            return_dict_in_generate = True,
        )
    finally:
        hook_manager.remove()

    # ── Trajectory entropy ──────────────────────────────────────
    # scores: tuple of (vocab_size,) log-softmax tensors per generated token
    if out.scores:
        logits_stack = torch.stack(out.scores, dim=0)   # (T, vocab)
        traj_entropy = float(entropy_from_logits(logits_stack))
    else:
        traj_entropy = float("inf")

    # ── Final hidden state ──────────────────────────────────────
    # hidden_states[-1][-1][0, -1, :]  → last layer, last token, first batch
    if out.hidden_states:
        final_hs = out.hidden_states[-1][-1][0, -1, :]   # (d_model,)
        final_hs = final_hs.to(cfg.hardware.adapter_device)
    else:
        # Fallback: re-run a forward pass to get hidden state
        final_ids = out.sequences
        with torch.no_grad():
            hook_manager.install(agent)
            try:
                fwd = model(
                    input_ids=final_ids,
                    output_hidden_states=True,
                )
            finally:
                hook_manager.remove()
        final_hs = fwd.hidden_states[-1][0, -1, :].to(cfg.hardware.adapter_device)

    # ── Decode ──────────────────────────────────────────────────
    generated_ids = out.sequences[0, prompt_ids.shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return text, traj_entropy, final_hs


# ─────────────────────────────────────────────────────────────────
# Utility: extract layer specs from base model
# ─────────────────────────────────────────────────────────────────

def get_layer_specs(
    model: AutoModelForCausalLM,
    target_suffixes: set,
) -> Dict[str, Tuple[int, int]]:
    """
    Walks the model and returns {full_layer_name: (d_in, d_out)} for each
    targeted Linear layer. Used to initialise all agents with matching shapes.
    """
    specs = {}
    for name, module in model.named_modules():
        if any(name.endswith(suf) for suf in target_suffixes):
            if isinstance(module, nn.Linear):
                specs[name] = (module.in_features, module.out_features)
    return specs
