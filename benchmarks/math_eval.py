"""
benchmarks/math_eval.py
MATH dataset (Hendrycks et al.) – levels 3, 4, 5 (hard problems).
Uses the standard \boxed{} answer format.
"""

from __future__ import annotations
import re
import logging
from typing import Dict, List

from datasets import load_dataset

from benchmarks.base_eval import BaseEvaluator, extract_boxed_answer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step by step, "
    "then write your final answer inside \\boxed{}. "
    "Be concise but rigorous."
)

_FEW_SHOT = """
Problem: What is the value of $\\sqrt{16}$?
Solution: $\\sqrt{16} = 4$ \\boxed{4}

Problem: Simplify $\\frac{3}{6}$.
Solution: $\\frac{3}{6} = \\frac{1}{2}$ \\boxed{\\frac{1}{2}}
""".strip()


class MATHEvaluator(BaseEvaluator):
    BENCHMARK_NAME = "math"
    LEVELS = {3, 4, 5}   # hard problems only

    def load_dataset(self) -> List[Dict]:
        ds = load_dataset("hendrycks/competition_math", split=self.split, trust_remote_code=True)
        samples = []
        for i, item in enumerate(ds):
            if int(item["level"].split()[-1]) not in self.LEVELS:
                continue
            samples.append({
                "id":       f"math_{i}",
                "question": item["problem"],
                "answer":   item["solution"],   # raw solution string
                "level":    item["level"],
                "type":     item["type"],
            })
        return samples

    def format_prompt(self, sample: Dict) -> str:
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Examples:\n{_FEW_SHOT}\n\n"
            f"Problem: {sample['question']}\n"
            f"Solution:"
        )

    def extract_answer(self, raw_output: str) -> str:
        return extract_boxed_answer(raw_output)

    def is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Compare using symbolic equivalence when possible,
        falling back to string normalization.
        """
        gt_boxed = extract_boxed_answer(ground_truth)
        return _math_equiv(predicted, gt_boxed)


# ─────────────────────────────────────────────────────────────────
# Symbolic math equivalence
# ─────────────────────────────────────────────────────────────────

def _normalize_expr(s: str) -> str:
    """Lightweight normalization for numeric answers."""
    s = s.strip().replace(" ", "").replace(",", "")
    # Remove trailing .0
    s = re.sub(r"\.0+$", "", s)
    return s.lower()


def _math_equiv(pred: str, gt: str) -> bool:
    """Return True if pred and gt are mathematically equivalent."""
    pred = _normalize_expr(pred)
    gt   = _normalize_expr(gt)

    if pred == gt:
        return True

    # Try numeric comparison
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except (ValueError, TypeError):
        pass

    # Try sympy for symbolic equivalence
    try:
        from sympy import simplify, sympify, SympifyError
        p_sym = sympify(pred.replace("^", "**"))
        g_sym = sympify(gt.replace("^", "**"))
        diff  = simplify(p_sym - g_sym)
        return diff == 0
    except Exception:
        pass

    return False
