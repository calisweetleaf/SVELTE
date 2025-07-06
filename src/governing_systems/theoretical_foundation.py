"""Tools for verifying theoretical assumptions used by SVELTE."""

from __future__ import annotations

import logging
from typing import List


class AxiomVerificationSystem:
    """Very lightweight axiom checker."""

    def __init__(self, axioms: List[str]) -> None:
        self.axioms = axioms
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def verify(self, text: str) -> List[str]:
        """Return axioms that appear in the provided text."""
        found = []
        lower = text.lower()
        for ax in self.axioms:
            if ax.lower() in lower:
                found.append(ax)
        self.logger.debug("Verified %d/%d axioms", len(found), len(self.axioms))
        return found
