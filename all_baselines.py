"""
baselines/all_baselines.py
All five baseline methods from the experiment design doc.

1. SingleAgentCoT           – greedy / self-consistency (no swarm)
2. TextDebate               – N agents, share full reasoning traces
3. SparseText               – N agents, share final answer only
4. StaticLoRAAverage        – N LoRAs merged uniformly every step (no entropy/TIES)
5. IndividualLoRAEnsemble   – N LoRAs, no merging; majority vote at end
"""

from __future__ import annotations
import logging
import random
from collections import Counter
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ECHOSConfig
from echos.agent import ECHOSAgent, generate_with_agent, LoRAHookManager, get_layer_specs
from echos.merging import decompose_to_lora

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Shared generation helper
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _greedy_generate(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cfg: ECHOSConfig,
    max_new_tokens: int | None = None,
) -> str:
    hw  = cfg.hardware
    gen = cfg.generation
    enc = tokenizer(prompt, return_tensors="pt").to(hw.base_model_device)
    out = model.generate(
        **enc,
        max_new_tokens   = max_new_tokens or gen.max_new_tokens,
        do_sample        = False,
        repetition_penalty = gen.repetition_penalty,
    )
    ids = out[0, enc["input_ids"].shape[-1]:]
    return tokenizer.decode(ids, skip_special_tokens=True)


@torch.no_grad()
def _sample_generate(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cfg: ECHOSConfig,
) -> str:
    hw  = cfg.hardware
    gen = cfg.generation
    enc = tokenizer(prompt, return_tensors="pt").to(hw.base_model_device)
    out = model.generate(
        **enc,
        max_new_tokens     = gen.max_new_tokens,
        do_sample          = True,
        temperature        = gen.temperature,
        top_p              = gen.top_p,
        repetition_penalty = gen.repetition_penalty,
    )
    ids = out[0, enc["input_ids"].shape[-1]:]
    return tokenizer.decode(ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────
# 1. Single-Agent CoT
# ─────────────────────────────────────────────────────────────────

class SingleAgentCoT:
    """
    Simple single-agent: either greedy or self-consistency (M samples, majority vote).
    """
    def __init__(self, model, tokenizer, cfg: ECHOSConfig, n_samples: int = 1):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.n_samples = n_samples   # 1 = greedy, >1 = self-consistency

    def solve(self, prompt: str) -> str:
        if self.n_samples == 1:
            return _greedy_generate(prompt, self.model, self.tokenizer, self.cfg)

        # Self-consistency: sample M times, majority vote
        answers = [
            _sample_generate(prompt, self.model, self.tokenizer, self.cfg)
            for _ in range(self.n_samples)
        ]
        counter = Counter(a.strip().lower() for a in answers)
        best    = counter.most_common(1)[0][0]
        return best


# ─────────────────────────────────────────────────────────────────
# 2. Text-Based Debate (majority voting, full trace sharing)
# ─────────────────────────────────────────────────────────────────

_DEBATE_ROUND_TEMPLATE = (
    "Other agents provided the following answers:\n"
    "{peer_answers}\n\n"
    "Given this, revise or confirm your answer:\n"
)


class TextDebate:
    """
    N agents generate initial answers, then share full traces for 1-2 rounds.
    Context grows as O(N*L); capped at max_context_tokens to avoid OOM.
    """
    def __init__(
        self,
        model, tokenizer, cfg: ECHOSConfig,
        n_agents: int = 15,
        n_rounds: int = 1,
        max_context_tokens: int = 4096,
    ):
        self.model              = model
        self.tokenizer          = tokenizer
        self.cfg                = cfg
        self.n_agents           = n_agents
        self.n_rounds           = n_rounds
        self.max_context_tokens = max_context_tokens

    def solve(self, prompt: str) -> str:
        gen = self.cfg.generation

        # Round 0: independent generation
        answers: List[str] = [
            _sample_generate(prompt, self.model, self.tokenizer, self.cfg)
            for _ in range(self.n_agents)
        ]

        # Debate rounds
        for _ in range(self.n_rounds):
            new_answers = []
            for i in range(self.n_agents):
                peers = [answers[j] for j in range(self.n_agents) if j != i]
                # Truncate peers to stay within context budget
                peer_block = "\n---\n".join(peers)
                peer_block = self.tokenizer.decode(
                    self.tokenizer.encode(peer_block)[: self.max_context_tokens // 2],
                    skip_special_tokens=True,
                )
                debate_prompt = (
                    prompt + "\n\nYour previous answer:\n" + answers[i] + "\n\n" +
                    _DEBATE_ROUND_TEMPLATE.format(peer_answers=peer_block)
                )
                new_answers.append(
                    _sample_generate(debate_prompt, self.model, self.tokenizer, self.cfg)
                )
            answers = new_answers

        # Majority vote on final answers
        counter = Counter(a.strip().lower() for a in answers)
        return counter.most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────
# 3. Sparse Text (share final answer only, no reasoning traces)
# ─────────────────────────────────────────────────────────────────

class SparseText:
    """
    Agents share only final answers (not full traces).
    Isolates value of parameter communication vs token communication.
    """
    def __init__(self, model, tokenizer, cfg: ECHOSConfig, n_agents: int = 15):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.n_agents  = n_agents

    def solve(self, prompt: str, extract_fn=None) -> str:
        answers = [
            _sample_generate(prompt, self.model, self.tokenizer, self.cfg)
            for _ in range(self.n_agents)
        ]
        if extract_fn:
            answers = [extract_fn(a) for a in answers]
        counter = Counter(a.strip().lower() for a in answers)
        return counter.most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────
# 4. Static LoRA Averaging (no entropy routing, no TIES)
# ─────────────────────────────────────────────────────────────────

class StaticLoRAAverage:
    """
    N LoRA adapters merged by simple mean every step.
    Tests the value of dynamic topology + TIES vs static averaging.
    """
    def __init__(self, model, tokenizer, cfg: ECHOSConfig):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        hw             = cfg.hardware
        lora_cfg       = cfg.lora
        sw             = cfg.swarm

        self.layer_specs    = get_layer_specs(model, set(lora_cfg.target_modules))
        self.agents: List[ECHOSAgent] = [
            ECHOSAgent(i, self.layer_specs, lora_cfg,
                       hw.adapter_device, hw.torch_dtype())
            for i in range(sw.n_agents)
        ]
        self.hook_manager = LoRAHookManager(
            model, set(lora_cfg.target_modules), hw.base_model_device
        )

    def solve(self, prompt: str) -> str:
        cfg = self.cfg
        sw  = cfg.swarm
        enc = self.tokenizer(prompt, return_tensors="pt")

        for _ in range(sw.max_steps):
            answers = []
            for agent in self.agents:
                text, _, _ = generate_with_agent(
                    agent, enc["input_ids"], self.model,
                    self.tokenizer, self.hook_manager, cfg,
                )
                answers.append(text)

            # Static mean merge: every agent gets the mean of all others
            for layer_name in self.layer_specs:
                all_deltas = [
                    ag.adapters[layer_name].delta() for ag in self.agents
                ]
                mean_delta = torch.stack(all_deltas).mean(dim=0)
                for ag in self.agents:
                    W_new = ag.adapters[layer_name].delta() + 0.5 * mean_delta
                    A_new, B_new = decompose_to_lora(W_new, cfg.lora.r)
                    ag.adapters[layer_name].set_matrices(A_new, B_new)

        counter = Counter(a.strip().lower() for a in answers)
        return counter.most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────
# 5. Individual LoRA Ensemble (no merging, vote at end)
# ─────────────────────────────────────────────────────────────────

class IndividualLoRAEnsemble:
    """
    N independent LoRA agents – no gossip, no merging.
    Majority vote over final answers.
    Isolates the benefit of test-time adaptation vs static ensemble.
    """
    def __init__(self, model, tokenizer, cfg: ECHOSConfig):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        hw      = cfg.hardware
        lora_cfg = cfg.lora
        sw       = cfg.swarm

        self.layer_specs = get_layer_specs(model, set(lora_cfg.target_modules))
        self.agents: List[ECHOSAgent] = [
            ECHOSAgent(i, self.layer_specs, lora_cfg,
                       hw.adapter_device, hw.torch_dtype())
            for i in range(sw.n_agents)
        ]
        self.hook_manager = LoRAHookManager(
            model, set(lora_cfg.target_modules), hw.base_model_device
        )

    def solve(self, prompt: str) -> str:
        cfg = self.cfg
        enc = self.tokenizer(prompt, return_tensors="pt")
        answers = []
        for agent in self.agents:
            text, _, _ = generate_with_agent(
                agent, enc["input_ids"], self.model,
                self.tokenizer, self.hook_manager, cfg,
            )
            answers.append(text)
        counter = Counter(a.strip().lower() for a in answers)
        return counter.most_common(1)[0][0]
