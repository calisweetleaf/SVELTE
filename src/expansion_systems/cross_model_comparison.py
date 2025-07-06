"""Utilities for comparing tensor fields from different models."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def _l2_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    if a.shape != b.shape:
        raise ValueError("Tensor shapes do not match")
    return float(np.linalg.norm(a - b))


class CrossModelComparisonSystem:
    """Simple cross-model comparison based on L2 distance."""

    def compare(self, model_a: Dict[str, np.ndarray], model_b: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Return L2 norm differences for common tensors."""
        results: Dict[str, float] = {}
        common = set(model_a).intersection(model_b)
        for name in common:
            results[name] = _l2_diff(model_a[name], model_b[name])
        return results

    def summary(self, model_a: Dict[str, np.ndarray], model_b: Dict[str, np.ndarray]) -> Tuple[int, float]:
        """Return number of compared tensors and average difference."""
        diffs = self.compare(model_a, model_b)
        if not diffs:
            return 0, 0.0
        return len(diffs), float(np.mean(list(diffs.values())))
