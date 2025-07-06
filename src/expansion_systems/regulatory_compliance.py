"""Regulatory compliance utilities."""

from __future__ import annotations

from typing import Dict, Callable, Any


class RegulatoryComplianceSystem:
    """Perform simple compliance checks on model metadata."""

    def check(self, metadata: Dict[str, Any], rules: Dict[str, Callable[[Dict[str, Any]], bool]]) -> Dict[str, bool]:
        """Evaluate rules against provided metadata."""
        results: Dict[str, bool] = {}
        for name, rule in rules.items():
            try:
                results[name] = bool(rule(metadata))
            except Exception:
                results[name] = False
        return results

    def generate_report(self, results: Dict[str, bool]) -> str:
        """Return a simple textual compliance report."""
        lines = [f"{k}: {'PASS' if v else 'FAIL'}" for k, v in results.items()]
        return "\n".join(lines)
