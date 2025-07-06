"""Counterfactual analysis tools for SVELTE."""

from __future__ import annotations

from typing import Callable, Dict, Any
import copy

import numpy as np


class CounterfactualAnalysisSystem:
    """Perform simple counterfactual evaluations on tensor fields."""

    def run(self, tensor_field: Dict[str, np.ndarray], modify: Callable[[Dict[str, np.ndarray]], None], evaluate: Callable[[Dict[str, np.ndarray]], Any]) -> Any:
        """Apply a modification to a copy of the tensor field and evaluate it."""
        modified = {k: v.copy() for k, v in tensor_field.items()}
        modify(modified)
        return evaluate(modified)
