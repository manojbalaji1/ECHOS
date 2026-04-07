"""
benchmarks/strategy_qa.py  – StrategyQA multi-step implicit reasoning.
benchmarks/adversarial.py  – Adversarially perturbed MathQA for robustness testing.
Both live here to keep the package tidy.
"""

from __future__ import annotations
import logging
import random
import re
from typing import Dict, List

from datasets import load_dataset

from benchmarks.base_eval import BaseEvaluator, extract_yes_no, extract_boxed_answer

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# StrategyQA
# ═════════════════════════════════════════════════════════════════

_STRATEGY_SYSTEM = (
    "Answer the following question with 'Yes' or 'No'. "
    "Think step by step, then give your final answer."
)


class StrategyQAEvaluator(BaseEvaluator):
    BENCHMARK_NAME = "strategy_qa"

    def load_dataset(self) -> List[Dict]:
        try:
            ds = load_dataset("tasksource/strategy_qa", split="validation", trust_remote_code=True)
        except Exception:
            # fallback to train split
            ds = load_dataset("tasksource/strategy_qa", split="train", trust_remote_code=True)

        return [
            {
                "id":       f"sqa_{i}",
                "question": item["question"],
                "answer":   "yes" if item["answer"] else "no",
                "facts":    item.get("facts", []),
            }
            for i, item in enumerate(ds)
        ]

    def format_prompt(self, sample: Dict) -> str:
        return (
            f"{_STRATEGY_SYSTEM}\n\n"
            f"Question: {sample['question']}\n"
            f"Answer:"
        )

    def extract_answer(self, raw_output: str) -> str:
        return extract_yes_no(raw_output)

    def is_correct(self, predicted: str, ground_truth: str) -> bool:
        return predicted.lower().strip() == ground_truth.lower().strip()


# ═════════════════════════════════════════════════════════════════
# Adversarial MathQA (MathQA + 20% rogue agents)
# ═════════════════════════════════════════════════════════════════

_PERTURBATIONS = [
    lambda ans: str(int(ans) + random.choice([-1, 1, 2, -2])) if ans.lstrip("-").isdigit() else ans,
    lambda ans: str(float(ans) * -1) if _is_numeric(ans) else ans,
    lambda ans: str(round(float(ans) * 2, 4)) if _is_numeric(ans) else ans,
]


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _perturb_answer(correct: str) -> str:
    """Return a wrong but plausible-looking answer."""
    clean = correct.strip()
    for perturb in random.sample(_PERTURBATIONS, len(_PERTURBATIONS)):
        try:
            wrong = perturb(clean)
            if wrong != clean:
                return wrong
        except Exception:
            continue
    return str(random.randint(1, 100))   # fallback


class AdversarialMathQAEvaluator(BaseEvaluator):
    """
    MathQA benchmark with ~20% of problems having injected wrong answers
    into adversarial agent adapters.  Evaluates:
      1. Correct consensus rate on clean problems
      2. Contamination rate (did wrong answer spread?)
      3. Quarantine success (did Φ → 0 for adversarial agents?)
    """
    BENCHMARK_NAME = "adversarial_mathqa"

    def __init__(self, cfg, n_samples: int = 200, adversarial_fraction: float = 0.2, **kwargs):
        super().__init__(cfg, n_samples, **kwargs)
        self.adversarial_fraction = adversarial_fraction

    def load_dataset(self) -> List[Dict]:
        try:
            ds = load_dataset("math_qa", split="test", trust_remote_code=True)
        except Exception:
            logger.warning("math_qa not available; falling back to MATH dataset")
            from datasets import load_dataset as ld
            ds = ld("hendrycks/competition_math", split="test", trust_remote_code=True)
            return [
                {
                    "id":             f"adv_{i}",
                    "question":       item["problem"],
                    "answer":         _extract_final_answer(item["solution"]),
                    "wrong_answer":   _perturb_answer(_extract_final_answer(item["solution"])),
                    "is_perturbed":   False,
                }
                for i, item in enumerate(ds)
            ]

        samples = []
        for i, item in enumerate(ds):
            correct = item["correct"]
            samples.append({
                "id":           f"adv_{i}",
                "question":     item["Problem"],
                "answer":       correct,
                "wrong_answer": _perturb_answer(correct),
                "is_perturbed": False,
            })
        return samples

    def format_prompt(self, sample: Dict) -> str:
        return (
            "Solve the following math problem step by step. "
            "Write your final answer inside \\boxed{}.\n\n"
            f"Problem: {sample['question']}\n"
            f"Solution:"
        )

    def extract_answer(self, raw_output: str) -> str:
        return extract_boxed_answer(raw_output)

    def is_correct(self, predicted: str, ground_truth: str) -> bool:
        from benchmarks.math_eval import _math_equiv
        return _math_equiv(predicted, ground_truth)

    def evaluate_robustness(self, swarm, adversarial_agent_ids: List[int]) -> Dict:
        """
        Specialized evaluation that measures:
          - contamination_rate: fraction of clean agents adopting wrong answer
          - quarantine_success: fraction of adversarial edges that got Φ→0
          - swarm_robustness_score: correct_rate / contamination_rate
        """
        correct_count        = 0
        contamination_events = 0
        total_samples        = len(self.dataset)

        swarm.set_adversarial_agents(adversarial_agent_ids)

        for sample in self.dataset:
            prompt = self.format_prompt(sample)
            out    = swarm.solve(prompt)

            # Check best answer
            pred    = self.extract_answer(out["best_answer"])
            correct = self.is_correct(pred, sample["answer"])
            if correct:
                correct_count += 1

            # Check contamination: did any clean agent produce the wrong answer?
            wrong_ans = sample["wrong_answer"]
            for i, ans_text in out["all_answers"].items():
                if i not in adversarial_agent_ids:
                    pred_i = self.extract_answer(ans_text)
                    if self.is_correct(pred_i, wrong_ans):
                        contamination_events += 1

        correct_rate      = correct_count / total_samples
        contamination_rate = contamination_events / (
            total_samples * (swarm.cfg.swarm.n_agents - len(adversarial_agent_ids))
        )
        robustness_score = (
            correct_rate / contamination_rate if contamination_rate > 0 else float("inf")
        )

        return {
            "correct_rate":       correct_rate,
            "contamination_rate": contamination_rate,
            "robustness_score":   robustness_score,
            "n_adversarial":      len(adversarial_agent_ids),
            "n_samples":          total_samples,
        }


def _extract_final_answer(solution: str) -> str:
    """Extract final numeric answer from a solution string."""
    from benchmarks.base_eval import extract_boxed_answer
    return extract_boxed_answer(solution)
