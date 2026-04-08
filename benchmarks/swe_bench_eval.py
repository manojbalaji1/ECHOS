"""
benchmarks/swe_bench_eval.py
SWE-bench Lite evaluator.

Uses the official `swebench` package for execution-based evaluation.
ECHOS generates patches; the harness applies them and runs test suites.

Install:  pip install swebench
"""

from __future__ import annotations
import json
import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SWE_SYSTEM_PROMPT = """You are an expert software engineer.
You will be given a GitHub issue and the relevant source code.
Your task is to produce a minimal, correct patch in unified diff format.

Important:
- Only modify what is strictly necessary.
- Output ONLY the patch (unified diff), nothing else.
- The patch must apply cleanly with `git apply`.
"""

_PATCH_TEMPLATE = """
<issue>
{issue}
</issue>

<file_context>
{file_context}
</file_context>

Produce the unified diff patch to resolve the issue:
"""


class SWEBenchEvaluator:
    """
    Wraps SWE-bench Lite for ECHOS experiments.
    Generates patches via the swarm and evaluates via harness.
    """
    BENCHMARK_NAME = "swe_bench_lite"

    def __init__(self, cfg, n_samples: int = 50):
        self.cfg       = cfg
        self.n_samples = n_samples
        self._dataset: Optional[List[Dict]] = None

    @property
    def dataset(self) -> List[Dict]:
        if self._dataset is None:
            self._dataset = self._load()
        return self._dataset

    def _load(self) -> List[Dict]:
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "princeton-nlp/SWE-bench_Lite",
                split="test",
                trust_remote_code=True,
            )
            samples = list(ds)
            import random; random.shuffle(samples)
            return samples[: self.n_samples]
        except Exception as e:
            logger.error(f"Failed to load SWE-bench: {e}")
            return []

    def format_prompt(self, sample: Dict) -> str:
        # Truncate file context to avoid OOM (SWE instances can be huge)
        file_ctx = sample.get("text", "")[:8000]
        return _SWE_SYSTEM_PROMPT + _PATCH_TEMPLATE.format(
            issue       = sample.get("problem_statement", ""),
            file_context = file_ctx,
        )

    def extract_patch(self, raw_output: str) -> str:
        """
        Extract unified diff from model output.
        Model may include markdown code fences.
        """
        import re
        # Try fenced block
        m = re.search(r"```(?:diff|patch)?\n(.*?)```", raw_output, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback: look for lines starting with --- or +++
        lines = raw_output.splitlines()
        patch_lines = []
        in_patch = False
        for line in lines:
            if line.startswith("---") or line.startswith("+++"):
                in_patch = True
            if in_patch:
                patch_lines.append(line)
        return "\n".join(patch_lines) if patch_lines else raw_output.strip()

    def evaluate_echos(self, swarm) -> Dict:
        """
        Run ECHOS on all SWE-bench samples and collect patches.
        Returns dict ready for swebench harness.
        """
        predictions = []
        for sample in self.dataset:
            prompt  = self.format_prompt(sample)
            out     = swarm.solve(prompt)
            patch   = self.extract_patch(out["best_answer"])
            predictions.append({
                "instance_id": sample["instance_id"],
                "model_patch": patch,
                "model_name_or_path": self.cfg.model_name,
            })
            logger.debug(f"  [{sample['instance_id']}] patch_len={len(patch)}")

        return {"predictions": predictions, "n_samples": len(predictions)}

    def run_harness(self, predictions: List[Dict], output_dir: str) -> Dict:
        """
        Invoke the official SWE-bench evaluation harness.
        Requires Docker + swebench installed.
        """
        pred_path = os.path.join(output_dir, "predictions.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(pred_path, "w") as f:
            json.dump(predictions, f)

        try:
            result = subprocess.run(
                [
                    "python", "-m", "swebench.harness.run_evaluation",
                    "--predictions_path", pred_path,
                    "--max_workers", "4",
                    "--run_id",       "echos_eval",
                ],
                capture_output=True, text=True, timeout=7200,
            )
            logger.info(result.stdout[-2000:])
            if result.returncode != 0:
                logger.error(result.stderr[-2000:])
                return {"error": result.stderr[-500:]}

            # Parse result JSON written by harness
            result_file = os.path.join(output_dir, "results.json")
            if os.path.exists(result_file):
                with open(result_file) as f:
                    return json.load(f)
        except subprocess.TimeoutExpired:
            logger.error("SWE-bench harness timed out")
        except FileNotFoundError:
            logger.warning(
                "swebench not installed or Docker not available. "
                "Returning predictions only; run harness manually."
            )
        return {"predictions_saved": pred_path}
