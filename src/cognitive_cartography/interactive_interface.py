"""Simple interactive interface for exploring SVELTE visualisations."""

from __future__ import annotations

import logging
from typing import List

from .visualization_engine import VisualizationEngine


class InteractiveInterface:
    """Command line interface for interactive exploration."""

    def __init__(self, engine: VisualizationEngine) -> None:
        self.engine = engine
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def start(self) -> None:
        """Begin the interactive loop."""
        tensors: List[str] = list(self.engine.tensor_field.keys())
        self.logger.info("Starting interactive interface")
        print("Available tensors:")
        for idx, name in enumerate(tensors):
            print(f"  [{idx}] {name}")
        print("Type the number of a tensor to visualise or 'q' to quit.")

        while True:
            choice = input("Selection: ").strip()
            if choice.lower() in {"q", "quit", "exit"}:
                break
            if choice.isdigit() and int(choice) < len(tensors):
                name = tensors[int(choice)]
                try:
                    self.engine.plot_tensor(name)
                except Exception as exc:  # pragma: no cover - visual output
                    self.logger.error("Failed to plot tensor %s: %s", name, exc)
            else:
                print("Invalid selection")

        self.logger.info("Interactive session finished")
