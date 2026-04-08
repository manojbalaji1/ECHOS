"""
benchmarks/base_eval.py
Abstract base class for all benchmark evaluators.
Handles prompt formatting, dataset loading, result aggregation,
and ties into the ECHOS swarm + baselines in a uniform API.
"""

from __future__ import annotations
import abc
import json
import logging
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    sample_id: str
    question: str
    ground_truth: str
    model_answer: str
    is_correct: bool
    extra: Dict = field(default_factory=dict)   # entropy, n_steps, etc.


@dataclass
class BenchmarkResult:
    benchmark: str
    method: str
    model_name: str
    n_samples: int
    accuracy: float
    std: float = 0.0
    samples: List[SampleResult] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class BaseEvaluator(abc.ABC):
    """
    Subclasses implement:
      - load_dataset()   → list of {id, question, answer} dicts
      - format_prompt()  → str
      - extract_answer() → str
      - is_correct()     → bool
    """

    BENCHMARK_NAME: str = "base"

    def __init__(self, cfg, n_samples: int = 500, split: str = "test"):
        self.cfg       = cfg
        self.n_samples = n_samples
        self.split     = split
        self._dataset: Optional[List[Dict]] = None

    @property
    def dataset(self) -> List[Dict]:
        if self._dataset is None:
            self._dataset = self.load_dataset()
            random.shuffle(self._dataset)
            if self.n_samples < len(self._dataset):
                self._dataset = self._dataset[: self.n_samples]
            logger.info(f"{self.BENCHMARK_NAME}: {len(self._dataset)} samples loaded")
        return self._dataset

    @abc.abstractmethod
    def load_dataset(self) -> List[Dict]:
        ...

    @abc.abstractmethod
    def format_prompt(self, sample: Dict) -> str:
        ...

    @abc.abstractmethod
    def extract_answer(self, raw_output: str) -> str:
        ...

    @abc.abstractmethod
    def is_correct(self, predicted: str, ground_truth: str) -> bool:
        ...

    # ─────────────────────────────────────────
    # Evaluation runners
    # ─────────────────────────────────────────

    def evaluate_echos(self, swarm, method_name: str = "echos") -> BenchmarkResult:
        """Run ECHOS swarm on all samples."""
        results = []
        for sample in self.dataset:
            prompt  = self.format_prompt(sample)
            out     = swarm.solve(prompt)
            raw     = out["best_answer"]
            pred    = self.extract_answer(raw)
            correct = self.is_correct(pred, sample["answer"])
            results.append(SampleResult(
                sample_id    = sample["id"],
                question     = sample["question"],
                ground_truth = sample["answer"],
                model_answer = pred,
                is_correct   = correct,
                extra        = {
                    "n_steps":         out["n_steps"],
                    "best_agent":      out["best_agent"],
                    "terminal_entropy": float(
                        out["entropy_history"][:, -1].min()
                        if out["entropy_history"].shape[1] > 0 else float("nan")
                    ),
                },
            ))
            logger.debug(f"  [{sample['id']}] pred={pred!r}  gt={sample['answer']!r}  ✓={correct}")

        return self._compile(results, method_name)

    def evaluate_baseline(
        self,
        model,
        tokenizer,
        baseline_fn,
        method_name: str,
    ) -> BenchmarkResult:
        """Run a baseline (function: prompt → str) on all samples."""
        results = []
        for sample in self.dataset:
            prompt  = self.format_prompt(sample)
            raw     = baseline_fn(prompt, model, tokenizer, self.cfg)
            pred    = self.extract_answer(raw)
            correct = self.is_correct(pred, sample["answer"])
            results.append(SampleResult(
                sample_id    = sample["id"],
                question     = sample["question"],
                ground_truth = sample["answer"],
                model_answer = pred,
                is_correct   = correct,
            ))
        return self._compile(results, method_name)

    def _compile(self, results: List[SampleResult], method: str) -> BenchmarkResult:
        acc = np.mean([r.is_correct for r in results])
        # Bootstrap std
        boots = [
            np.mean(random.choices([r.is_correct for r in results], k=len(results)))
            for _ in range(1000)
        ]
        std = float(np.std(boots))
        return BenchmarkResult(
            benchmark  = self.BENCHMARK_NAME,
            method     = method,
            model_name = self.cfg.model_name,
            n_samples  = len(results),
            accuracy   = float(acc),
            std        = std,
            samples    = results,
        )


# ─────────────────────────────────────────────────────────────────
# Answer extraction helpers (shared across benchmarks)
# ─────────────────────────────────────────────────────────────────

def extract_boxed_answer(text: str) -> str:
    """Extract \boxed{...} LaTeX answer from model output."""
    import re
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    # Fallback: last number-like token
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else text.strip()


def extract_multiple_choice(text: str) -> str:
    """Extract A/B/C/D answer from multiple choice output."""
    import re
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else text.strip().upper()[:1]


def extract_yes_no(text: str) -> str:
    """Extract yes/no from binary question output."""
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    # Search for first occurrence
    import re
    m = re.search(r"\b(yes|no)\b", t)
    return m.group(1) if m else t[:3]
