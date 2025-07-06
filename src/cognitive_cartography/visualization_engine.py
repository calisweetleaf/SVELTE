"""Visualization engine for SVELTE cognitive cartography.

This module provides minimal functionality for transforming tensors into
visual representations.  It supports basic dimensionality reduction via PCA
and simple plotting with matplotlib.  The goal is to offer an accessible
interface for interactive exploration while remaining lightweight for tests.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


class VisualizationEngine:
    """Create visualisations of tensor fields."""

    def __init__(self, tensor_field: Dict[str, np.ndarray], log_level: int = logging.INFO) -> None:
        self.tensor_field = tensor_field
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)
        self.logger.debug("VisualizationEngine initialised with %d tensors", len(tensor_field))

    def _reduce(self, data: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduce dimensionality of data using PCA."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(data)
        self.logger.debug("Reduced data from shape %s to %s", data.shape, reduced.shape)
        return reduced

    def plot_tensor(self, name: str, n_components: int = 2, save_path: Optional[str] = None) -> None:
        """Plot a tensor after dimensionality reduction."""
        if name not in self.tensor_field:
            raise KeyError(f"Tensor '{name}' not found")

        data = self.tensor_field[name]
        reduced = self._reduce(data, n_components=n_components)

        plt.figure(figsize=(6, 5))
        if n_components == 1:
            plt.plot(reduced[:, 0])
        elif n_components == 2:
            plt.scatter(reduced[:, 0], reduced[:, 1], s=10)
        else:
            ax = plt.axes(projection="3d")
            ax.scatter3D(reduced[:, 0], reduced[:, 1], reduced[:, 2], s=10)

        plt.title(f"Tensor: {name}")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            self.logger.info("Saved visualisation of %s to %s", name, save_path)
        else:
            plt.show()
        plt.close()
