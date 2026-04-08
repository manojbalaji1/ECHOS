"""
benchmarks/gpqa_eval.py
GPQA Diamond subset – graduate-level expert questions in biology, chemistry, physics.
4-choice multiple choice.  Uses the "gpqa" dataset on HuggingFace.
"""

from __future__ import annotations
import logging
import random
from typing import Dict, List

from datasets import load_dataset

from benchmarks.base_eval import BaseEvaluator, extract_multiple_choice

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert scientist. Read the question carefully, "
    "reason step by step, then state the answer as a single letter (A, B, C, or D)."
)


class GPQAEvaluator(BaseEvaluator):
    BENCHMARK_NAME = "gpqa_diamond"

    def load_dataset(self) -> List[Dict]:
        # Dataset: Idavidrein/gpqa  – "gpqa_diamond" subset
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load GPQA: {e}")
            return []

        samples = []
        for i, item in enumerate(ds):
            # Shuffle choices to avoid position bias
            choices = [
                item["Correct Answer"],
                item["Incorrect Answer 1"],
                item["Incorrect Answer 2"],
                item["Incorrect Answer 3"],
            ]
            letters = ["A", "B", "C", "D"]
            order   = list(range(4))
            random.shuffle(order)
            shuffled   = [choices[o] for o in order]
            correct_idx = order.index(0)   # index of correct answer after shuffle
            correct_letter = letters[correct_idx]

            choices_str = "\n".join(
                f"{letters[k]}. {shuffled[k]}" for k in range(4)
            )
            samples.append({
                "id":       f"gpqa_{i}",
                "question": item["Question"],
                "choices":  choices_str,
                "answer":   correct_letter,
                "domain":   item.get("High-level domain", ""),
            })
        return samples

    def format_prompt(self, sample: Dict) -> str:
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Question: {sample['question']}\n\n"
            f"{sample['choices']}\n\n"
            f"Answer (A/B/C/D):"
        )

    def extract_answer(self, raw_output: str) -> str:
        return extract_multiple_choice(raw_output)

    def is_correct(self, predicted: str, ground_truth: str) -> bool:
        return predicted.upper().strip() == ground_truth.upper().strip()
