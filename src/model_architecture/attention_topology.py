"""
Attention Topology System for SVELTE Framework.
Analyzes attention as differential geometric structures and computes curvature tensors.

This module interprets attention mechanisms through differential geometry, enabling
the analysis of information flow as a geometric phenomenon on manifolds.

author: Morpheus
date: 2025-05-01
version: 0.1.0
description: This module provides a system for analyzing attention mechanisms as geometric manifolds.
ID: 001
SHA-256: 1234567890abcdef1234567890abcdef12345678
"""

import numpy as np
import logging
import json
import argparse
import sys
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
try:
    from scipy import stats as scipy_stats
    SCIPY_STATS_AVAILABLE = True
except ImportError:
    SCIPY_STATS_AVAILABLE = False
from enum import Enum
import warnings

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class CurvatureMethod(Enum):
 """Methods for computing manifold curvature."""
 RIEMANN = "riemann"
 RICCI = "ricci"
 SCALAR = "scalar"
 SECTIONAL = "sectional"


@dataclass
class TopologyMetrics:
 """Container for topology analysis metrics."""
 curvature: np.ndarray
 entropy: float
 homology: Dict[str, Any]
 geodesics: Optional[np.ndarray] = None
 eigenvectors: Optional[np.ndarray] = None
 eigenvalues: Optional[np.ndarray] = None


class AttentionTopologySystem:
 """
 Analyzes attention mechanisms as geometric manifolds.
 
 This system interprets attention weight matrices as defining Riemannian manifolds
 and computes various differential geometric quantities to characterize the
 information flow properties of the attention mechanism.
 """
 
 def __init__(self, 
     tensor_field: Dict[str, np.ndarray],
     eps: float = 1e-10,
     metric_type: str = "euclidean",
     log_level: int = logging.INFO):
  """
  Initialize the attention topology analyzer.
  
  Args:
   tensor_field: Dictionary mapping layer names to attention tensors
   eps: Small constant to ensure numerical stability
   metric_type: Type of distance metric to use for local geometry
   log_level: Logging verbosity level
  """
  self._setup_logging(log_level)
  self._validate_inputs(tensor_field)
  
  self.tensor_field = tensor_field
  self.eps = eps
  self.metric_type = metric_type
  self.curvature_tensors = {}
  self.metric_tensors = {}
  self.christoffel_symbols = {}
  self.cached_results = {}
  
  self.logger.info(f"Initialized AttentionTopologySystem with {len(tensor_field)} tensor fields")
  self.SCIPY_STATS_AVAILABLE = SCIPY_STATS_AVAILABLE # Store for local use
  self.MATPLOTLIB_AVAILABLE = MATPLOTLIB_AVAILABLE # Store for local use

 def _setup_logging(self, log_level: int) -> None:
  """Configure logging for the topology system."""
  self.logger = logging.getLogger("AttentionTopology")
  if not self.logger.handlers:
   handler = logging.StreamHandler()
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   handler.setFormatter(formatter)
   self.logger.addHandler(handler)
  self.logger.setLevel(log_level)

 def _validate_inputs(self, tensor_field: Dict[str, np.ndarray]) -> None:
  """
  Validate input tensor fields.
  
  Args:
   tensor_field: Dictionary of attention tensors to validate
   
  Raises:
   ValueError: If tensors are invalid or improperly shaped
  """
  if not tensor_field:
   raise ValueError("Tensor field dictionary cannot be empty")
   
  for name, tensor in tensor_field.items():
   if not isinstance(tensor, np.ndarray):
    raise TypeError(f"Tensor {name} must be a numpy array, got {type(tensor)}")
   
   if tensor.ndim < 2:
    raise ValueError(f"Tensor {name} must have at least 2 dimensions")
    
   if np.isnan(tensor).any() or np.isinf(tensor).any():
    raise ValueError(f"Tensor {name} contains NaN or Inf values")

 def compute_metric_tensor(self, tensor: np.ndarray) -> np.ndarray:
  """
  Compute the metric tensor from an attention tensor.
  
  Args:
   tensor: Attention weight tensor
   
  Returns:
   The metric tensor representing local geometry
  """
  # For attention weights, we'll use a metric that captures relationships between positions
  # Shape: batch_size × seq_len × seq_len
  if tensor.ndim == 4:  # Multi-head attention: batch_size × num_heads × seq_len × seq_len
   # Average across attention heads
   tensor = np.mean(tensor, axis=1)
   
  # Create metric tensor using the attention weights directly
  # g_ij = -log(A_ij + eps) to convert attention weights to distances
  metric = -np.log(tensor + self.eps) # metric can be (batch, dim, dim) or (dim, dim)

  # Ensure the metric is positive definite.
  if metric.ndim == 3: # Batched
      for b_idx in range(metric.shape[0]):
          g_slice = metric[b_idx]
          try:
              # Adding a small epsilon to the diagonal before eigvalsh for stability
              # g_stable = g_slice + np.eye(g_slice.shape[0]) * self.eps
              min_eig_val = np.min(np.linalg.eigvalsh(g_slice))
              if min_eig_val <= self.eps: # Check for non-positive definiteness (use eps for tolerance)
                  self.logger.warning(f"Adjusting metric tensor for batch {b_idx} to ensure positive definiteness. Min eigenvalue: {min_eig_val:.2e}")
                  metric[b_idx] += (np.abs(min_eig_val) + self.eps) * np.eye(g_slice.shape[-1])
          except np.linalg.LinAlgError:
              self.logger.warning(f"LinAlgError for batch {b_idx} during eigvalsh for positive definiteness check. Adding epsilon * I.")
              metric[b_idx] += self.eps * np.eye(g_slice.shape[-1]) # Add epsilon * Identity
  elif metric.ndim == 2: # Single
      try:
          # g_stable = metric + np.eye(metric.shape[0]) * self.eps
          min_eig_val = np.min(np.linalg.eigvalsh(metric))
          if min_eig_val <= self.eps: # Check for non-positive definiteness
              self.logger.warning(f"Adjusting metric tensor to ensure positive definiteness. Min eigenvalue: {min_eig_val:.2e}")
              metric += (np.abs(min_eig_val) + self.eps) * np.eye(metric.shape[-1])
      except np.linalg.LinAlgError:
          self.logger.warning("LinAlgError during eigvalsh for positive definiteness check. Adding epsilon * I.")
          metric += self.eps * np.eye(metric.shape[-1]) # Add epsilon * Identity
  return metric

 def compute_christoffel_symbols(self, metric: np.ndarray) -> np.ndarray:
  """
  Compute the Christoffel symbols (Γ^k_ij) from the metric tensor (g_ij).
  Formula: Γ^k_ij = 0.5 * g^{km} (∂_i g_{mj} + ∂_j g_{mi} - ∂_m g_{ij})
  
  Args:
   metric: The metric tensor, shape (dim, dim) or (batch_size, dim, dim).
   
  Returns:
   Christoffel symbols of the second kind, shape (dim, dim, dim) or (batch_size, dim, dim, dim).
   Order: Γ^k_ij -> christoffel[b, k, i, j] (for batch) or christoffel[k,i,j] (single)
  """
  is_batched = metric.ndim == 3
  metric_batched = metric if is_batched else np.expand_dims(metric, axis=0)
  
  batch_size, dim, _ = metric_batched.shape
  christoffel_batched = np.zeros((batch_size, dim, dim, dim))

  for b in range(batch_size):
    g = metric_batched[b]
    try:
        g_inv = np.linalg.inv(g) # g^{km}
    except np.linalg.LinAlgError:
        self.logger.warning(f"Metric tensor for batch {b} is singular. Using pseudo-inverse.")
        g_inv = np.linalg.pinv(g)

    # metric_derivs_numerical[l, i, j] stores ∂_l g_ij (derivative of g_ij w.r.t. x^l)
    # This assumes that the coordinates x^l are aligned with the axes of the metric tensor g.
    # If g is (dim, dim), np.gradient(g) returns a list of 2 arrays (derivatives along axis 0 and 1).
    # If manifold dimension ('dim') is > g.ndim, derivatives along other axes are implicitly zero.
    metric_derivs_numerical = np.zeros((dim, dim, dim))
    
    grad_g_components = []
    if g.ndim >= 1 : grad_g_components.append(np.gradient(g, axis=0))
    if g.ndim >= 2 : grad_g_components.append(np.gradient(g, axis=1))
    # For higher g.ndim, extend this. Typically g is 2D (dim x dim matrix).

    for l_axis in range(len(grad_g_components)): # iterates 0, 1 for a 2D metric tensor
        metric_derivs_numerical[l_axis, :, :] = grad_g_components[l_axis]
    # For l_axis >= len(grad_g_components), metric_derivs_numerical[l_axis,:,:] remains zero.
    # This means ∂_l g_ij = 0 if l >= g.ndim.

    for k_christ in range(dim): # Index for Γ^k_ij
        for i_christ in range(dim): # Index for Γ^k_ij
            for j_christ in range(dim): # Index for Γ^k_ij
                sum_val = 0
                # Sum over m_summation for g^{k_christ, m_summation}
                for m_summation in range(dim):
                    # ∂_i g_{mj} -> metric_derivs_numerical[i_christ, m_summation, j_christ]
                    # ∂_j g_{mi} -> metric_derivs_numerical[j_christ, m_summation, i_christ]
                    # ∂_m g_{ij} -> metric_derivs_numerical[m_summation, i_christ, j_christ]

                    term1 = metric_derivs_numerical[i_christ, m_summation, j_christ]
                    term2 = metric_derivs_numerical[j_christ, m_summation, i_christ]
                    term3 = metric_derivs_numerical[m_summation, i_christ, j_christ]

                    sum_val += 0.5 * g_inv[k_christ, m_summation] * (term1 + term2 - term3)
                christoffel_batched[b, k_christ, i_christ, j_christ] = sum_val

  return christoffel_batched if is_batched else christoffel_batched[0]

 def compute_riemann_tensor(self, christoffel: np.ndarray) -> np.ndarray:
  """
  Compute the Riemann curvature tensor (R^i_jkl) from Christoffel symbols (Γ^i_jk).
  Formula: R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^i_mk Γ^m_jl - Γ^i_ml Γ^m_jk
  
  Args:
   christoffel: Christoffel symbols, shape (dim, dim, dim) or (batch_size, dim, dim, dim)
                Assumed order: Γ^i_jk -> christoffel[b, i, j, k] (for batch) or christoffel[i,j,k]
   
  Returns:
   Riemann curvature tensor, shape (dim,dim,dim,dim) or (batch_size,dim,dim,dim,dim)
   Order: R^i_jkl -> riemann[b, i, j, k, l] (for batch) or riemann[i,j,k,l]
  """
  is_batched = christoffel.ndim == 4
  christoffel_batched = christoffel if is_batched else np.expand_dims(christoffel, axis=0)
  
  # christoffel_batched has shape (batch_size, dim_Gamma_i, dim_Gamma_j, dim_Gamma_k)
  batch_size, dim, _, _ = christoffel_batched.shape
  riemann_batched = np.zeros((batch_size, dim, dim, dim, dim))

  for b in range(batch_size):
    symbols_Gamma_ijk = christoffel_batched[b] # Γ^i_jk (indices i,j,k)

    # Derivatives of Christoffel symbols: ∂_p Γ^i_jk
    # deriv_Gamma_numerical[p, i, j, k] = ∂_p Γ^i_jk
    # p is limited by symbols_Gamma_ijk.ndim (which is 3).
    deriv_Gamma_numerical = np.zeros((dim, dim, dim, dim)) # p, i, j, k
    
    grad_symbols_components = []
    if symbols_Gamma_ijk.ndim >= 1: grad_symbols_components.append(np.gradient(symbols_Gamma_ijk, axis=0))
    if symbols_Gamma_ijk.ndim >= 2: grad_symbols_components.append(np.gradient(symbols_Gamma_ijk, axis=1))
    if symbols_Gamma_ijk.ndim >= 3: grad_symbols_components.append(np.gradient(symbols_Gamma_ijk, axis=2))

    for p_axis in range(len(grad_symbols_components)): # 0, 1, 2 for a 3D Christoffel tensor
        deriv_Gamma_numerical[p_axis, :, :, :] = grad_symbols_components[p_axis]
    # For p_axis >= len(grad_symbols_components), derivatives are zero.

    for i_riem in range(dim): # R^i_jkl
        for j_riem in range(dim): # R^i_jkl
            for k_riem in range(dim): # R^i_jkl
                for l_riem in range(dim): # R^i_jkl
                    # Term ∂_k Γ^i_jl: derivative of Γ^i_jl w.r.t x^k (k_riem)
                    # Γ^i_jl is symbols_Gamma_ijk[i_riem, j_riem, l_riem]
                    # Its derivative w.r.t. x^k_riem is deriv_Gamma_numerical[k_riem, i_riem, j_riem, l_riem]
                    term_deriv1 = deriv_Gamma_numerical[k_riem, i_riem, j_riem, l_riem]

                    # Term ∂_l Γ^i_jk: derivative of Γ^i_jk w.r.t x^l (l_riem)
                    # Γ^i_jk is symbols_Gamma_ijk[i_riem, j_riem, k_riem]
                    # Its derivative w.r.t. x^l_riem is deriv_Gamma_numerical[l_riem, i_riem, j_riem, k_riem]
                    term_deriv2 = deriv_Gamma_numerical[l_riem, i_riem, j_riem, k_riem]

                    term_prod = 0
                    for m_summation in range(dim):
                        # Γ^i_mk * Γ^m_jl -> symbols_Gamma_ijk[i_riem, m_summation, k_riem] * symbols_Gamma_ijk[m_summation, j_riem, l_riem]
                        prod1 = symbols_Gamma_ijk[i_riem, m_summation, k_riem] * symbols_Gamma_ijk[m_summation, j_riem, l_riem]
                        # Γ^i_ml * Γ^m_jk -> symbols_Gamma_ijk[i_riem, m_summation, l_riem] * symbols_Gamma_ijk[m_summation, j_riem, k_riem]
                        prod2 = symbols_Gamma_ijk[i_riem, m_summation, l_riem] * symbols_Gamma_ijk[m_summation, j_riem, k_riem]
                        term_prod += (prod1 - prod2)

                    riemann_batched[b, i_riem, j_riem, k_riem, l_riem] = term_deriv1 - term_deriv2 + term_prod

  return riemann_batched if is_batched else riemann_batched[0]


 def compute_curvature(self, method: CurvatureMethod = CurvatureMethod.RIEMANN) -> Dict[str, np.ndarray]:
  """
  Compute curvature tensors for all attention layers.
  
  Args:
   method: Type of curvature to compute (Riemann, Ricci, scalar, sectional)
   
  Returns:
   Dictionary mapping layer names to curvature tensors
  """
  self.logger.info(f"Computing {method.value} curvature for {len(self.tensor_field)} layers")
  
  for name, tensor in self.tensor_field.items():
   self.logger.debug(f"Processing layer: {name}")
   
   # Compute the metric tensor from attention weights
   self.metric_tensors[name] = self.compute_metric_tensor(tensor)
   
   # Compute Christoffel symbols from the metric tensor
   self.christoffel_symbols[name] = self.compute_christoffel_symbols(self.metric_tensors[name])
   
   # Compute Riemann curvature tensor
   riemann = self.compute_riemann_tensor(self.christoffel_symbols[name])
   
   # Process based on the requested curvature type
   if method == CurvatureMethod.RIEMANN:
    self.curvature_tensors[name] = riemann
   elif method == CurvatureMethod.RICCI:
    # Ricci tensor is the contraction of the Riemann tensor
    ricci = np.trace(riemann, axis1=1, axis2=3)
    self.curvature_tensors[name] = ricci
   elif method == CurvatureMethod.SCALAR:
    # Scalar curvature is the trace of the Ricci tensor
    ricci = np.trace(riemann, axis1=1, axis2=3)
    scalar = np.trace(ricci, axis1=-2, axis2=-1)
    self.curvature_tensors[name] = scalar
   elif method == CurvatureMethod.SECTIONAL:
    # Compute sectional curvature K(u,v) = R(u,v,v,u) / (g(u,u)g(v,v) - g(u,v)^2)
    # For basis vectors e_i, e_j: K(e_i,e_j) = R_ijij / (g_ii g_jj - g_ij^2)
    # Riemann tensor (output of compute_riemann_tensor) is R^a_bcd.
    # We need R_abcd = g_ae R^e_bcd. For K(e_i,e_j), this means R_ijij.
    # R_ijij = g_ia R^a_jij (sum over 'a').
    
    current_metric = self.metric_tensors[name] # g_ij
    is_riemann_batched = riemann.ndim == 5
    is_metric_batched = current_metric.ndim == 3

    # Ensure consistent batching for metric if riemann is batched
    current_metric_batched = current_metric
    if is_riemann_batched and not is_metric_batched:
        current_metric_batched = np.tile(current_metric, (riemann.shape[0], 1, 1))
    elif not is_riemann_batched and is_metric_batched:
        # This case should ideally not happen if inputs are consistent
        self.logger.warning("Riemann tensor not batched but metric is. Using first metric slice.")
        current_metric_batched = current_metric_batched[0] # Make it non-batched
        is_metric_batched = False


    if is_riemann_batched:
        batch_size_r, dim_r, _, _, _ = riemann.shape # R^a_bcd -> riemann[b,a,b,c,d]
        sectional_batched = np.zeros((batch_size_r, dim_r * (dim_r - 1) // 2))

        for b_idx in range(batch_size_r):
            g = current_metric_batched[b_idx]       # g_ij for this batch item
            riemann_slice = riemann[b_idx]          # R^a_bcd for this batch item

            # R_ajlm = g_ak R^k_jlm (Einstein sum over k)
            # riemann_slice is R^k_jlm (k is first index, then j,l,m)
            # So, R_ijij = g_ia R^a_jij (sum over a)
            # r_abcd_cov[j_riem,i_plane,j_plane,i_plane,j_plane]
            # R_ijij = g_ia * R^a_jij where R^a_jij is riemann_slice[a,j,i,j]

            plane_idx = 0
            for i_plane in range(dim_r): # Index for e_i
                for j_plane in range(i_plane + 1, dim_r): # Index for e_j
                    # Numerator: R_ijij = g_ia R^a_jij (sum over 'a')
                    # riemann_slice[a, j_plane, i_plane, j_plane] is R^a_jij
                    numerator = np.sum(g[i_plane, :] * riemann_slice[:, j_plane, i_plane, j_plane])

                    denominator = (g[i_plane, i_plane] * g[j_plane, j_plane] -
                                   g[i_plane, j_plane] * g[i_plane, j_plane])

                    sectional_batched[b_idx, plane_idx] = numerator / (denominator + self.eps)
                    plane_idx += 1
        self.curvature_tensors[name] = sectional_batched if is_metric_batched else sectional_batched[0]
    
    else: # Non-batched Riemann R^a_bcd
        dim_r = riemann.shape[0] # R^a_bcd -> riemann[a,b,c,d]
        g = current_metric # Should be (dim_r, dim_r)
        sectional_single = np.zeros((dim_r * (dim_r - 1) // 2,))

        plane_idx = 0
        for i_plane in range(dim_r):
            for j_plane in range(i_plane + 1, dim_r):
                # Numerator R_ijij = g_ia R^a_jij (sum over 'a')
                # riemann[a, j_plane, i_plane, j_plane] is R^a_jij
                numerator = np.sum(g[i_plane, :] * riemann[:, j_plane, i_plane, j_plane])

                denominator = (g[i_plane, i_plane] * g[j_plane, j_plane] -
                               g[i_plane, j_plane] * g[i_plane, j_plane])
                sectional_single[plane_idx] = numerator / (denominator + self.eps)
                plane_idx += 1
        self.curvature_tensors[name] = sectional_single
   
   self.logger.debug(f"Completed curvature calculation for {name}")
   
  return self.curvature_tensors

 def analyze_topology(self, layer_name: str) -> TopologyMetrics:
  """
  Perform comprehensive topological analysis for a specific layer.
  
  Args:
   layer_name: Name of the layer to analyze
   
  Returns:
   TopologyMetrics containing analysis results
   
  Raises:
   KeyError: If the layer name is not found
  """
  if layer_name not in self.tensor_field:
   raise KeyError(f"Layer {layer_name} not found in tensor field")
   
  if layer_name not in self.curvature_tensors:
   self.logger.info(f"Computing curvature for {layer_name} (not previously calculated)")
   self.compute_curvature()
   
  # Extract tensors for this layer
  attention = self.tensor_field[layer_name]
  curvature = self.curvature_tensors[layer_name]
  
  # Compute attention entropy (measure of focus vs. diffusion)
  if attention.ndim == 4:  # Multi-head attention
   attention_avg = np.mean(attention, axis=1)
  else:
   attention_avg = attention
   
  entropy = -np.sum(attention_avg * np.log(attention_avg + self.eps), axis=-1).mean()
  
  # Compute persistent homology (topological features at different scales)
  # This is a simplified approximation
  homology = self._compute_homology(attention_avg)
  
  # Compute eigendecomposition of the curvature for principal directions
  if curvature.ndim <= 3:  # Only for Ricci or lower
   eigenvalues, eigenvectors = np.linalg.eigh(curvature)
  else:
   eigenvalues, eigenvectors = None, None
   
  # Return comprehensive metrics
  return TopologyMetrics(
   curvature=curvature,
   entropy=entropy,
   homology=homology,
   geodesics=self._compute_geodesics(self.metric_tensors[layer_name]),
   eigenvalues=eigenvalues,
   eigenvectors=eigenvectors
  )
 
 def _compute_homology(self, attention: np.ndarray) -> Dict[str, Any]:
  """
  Compute a simplified version of persistent homology.
  
  Args:
   attention: Attention weight matrix
   
  Returns:
   Dictionary with homology metrics
  """
  # Convert attention to distance matrix
  distance = 1 - attention
  
  # Simple Betti numbers approximation at different thresholds
  thresholds = [0.25, 0.5, 0.75]
  betti_0 = []  # Connected components
  betti_1 = []  # Loops/holes
  
  for thresh in thresholds:
   # Threshold the distance matrix to create a binary adjacency matrix
   adj = (distance < thresh).astype(int)
   
   # Count connected components (simplified)
   n_components, labels = connected_components(sp.csr_matrix(adj))
   betti_0.append(n_components)
   
   # Approximate number of loops (simplified)
   # In a complete graph, loops = edges - vertices + components
   edges = np.sum(adj) // 2  # Divide by 2 as each edge is counted twice
   vertices = adj.shape[0]
   betti_1.append(max(0, edges - vertices + n_components))
  
  return {
   "betti_0": betti_0,
   "betti_1": betti_1,
   "thresholds": thresholds
  }
 
 def _compute_geodesics(self, metric: np.ndarray) -> np.ndarray:
  """
  Compute geodesic distances between points on the attention manifold.
  
  Args:
   metric: Metric tensor defining the local geometry
   
  Returns:
   Matrix of geodesic distances
  """
  # For simplicity, approximate geodesics using the metric directly
  # In a full implementation, this would use the shortest path algorithm
  # with the metric determining local distances
  
  if metric.ndim > 2:
   # Handle batch dimension
   batch_size = metric.shape[0]
   dim = metric.shape[-1]
   geodesics = np.zeros((batch_size, dim, dim))
   
   for b in range(batch_size):
    # Use the Floyd-Warshall algorithm for geodesic distances
    dist = metric[b].copy()
    for k in range(dim):
     for i in range(dim):
      for j in range(dim):
       if dist[i, k] + dist[k, j] < dist[i, j]:
        dist[i, j] = dist[i, k] + dist[k, j]
    geodesics[b] = dist
  else:
   dim = metric.shape[-1]
   geodesics = metric.copy()
   
   # Floyd-Warshall
   for k in range(dim):
    for i in range(dim):
     for j in range(dim):
      if geodesics[i, k] + geodesics[k, j] < geodesics[i, j]:
       geodesics[i, j] = geodesics[i, k] + geodesics[k, j]
       
  return geodesics
  
 def visualize_curvature(self, layer_name: str, output_path: Optional[str] = None) -> None:
  """
  Visualize the curvature tensor for a specified layer.
  
  Args:
   layer_name: Name of the layer to visualize
   output_path: Path to save the visualization (if None, just displays)
   
  Raises:
   ImportError: If visualization dependencies are not available
   KeyError: If the layer name is not found
  """
  if not MATPLOTLIB_AVAILABLE:
   self.logger.error("Visualization requires matplotlib to be installed")
   raise ImportError("Please install matplotlib to use visualization features")
   
  if layer_name not in self.curvature_tensors:
   raise KeyError(f"Layer {layer_name} not found in curvature tensors")
   
  curvature = self.curvature_tensors[layer_name]
  
  # Create visualization based on curvature tensor dimensions
  fig = plt.figure(figsize=(12, 10))
  
  if curvature.ndim <= 2:  # Scalar curvature
   plt.imshow(curvature, cmap='viridis')
   plt.colorbar(label='Scalar Curvature')
   plt.title(f'Scalar Curvature for {layer_name}')
   
  elif curvature.ndim == 3:  # Ricci curvature
   # Use proper 3D axes type
   ax = fig.add_subplot(111, projection='3d')
   x, y = np.meshgrid(range(curvature.shape[0]), range(curvature.shape[1]))
   
   # Handle the case where curvature is 3D but we need 2D surface
   if curvature.shape[0] == curvature.shape[1]:
    # Use the diagonal or a specific slice for visualization
    surface_data = np.diagonal(curvature, axis1=0, axis2=1)
    if surface_data.ndim == 1:
     # Convert to 2D for surface plot
     surface_data = np.outer(surface_data, surface_data)
   else:
    surface_data = curvature[0] if curvature.shape[0] > 0 else curvature
   
   # Ensure we have a proper 2D surface
   if surface_data.ndim == 2:
    x, y = np.meshgrid(range(surface_data.shape[0]), range(surface_data.shape[1]))
    # Cast to 3D axes for proper method access
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(fig)
    ax.plot_surface(x, y, surface_data, cmap='viridis')
    ax.set_title(f'Ricci Curvature for {layer_name}')
    ax.set_xlabel('Dimension i')
    ax.set_ylabel('Dimension j')
    ax.set_zlabel('Curvature')
   else:
    # Fall back to 2D heatmap
    plt.imshow(curvature.reshape(curvature.shape[0], -1), cmap='viridis')
    plt.colorbar(label='Curvature')
    plt.title(f'Curvature Heatmap for {layer_name}')
   
  else:  # Higher order tensors
   # For high-dimensional tensors, show a 2D slice
   slice_idx = curvature.shape[0] // 2
   if curvature.ndim >= 3:
    slice_data = curvature[slice_idx]
    if slice_data.ndim > 2:
     # Further reduce dimensions
     slice_data = slice_data.reshape(slice_data.shape[0], -1)
   else:
    slice_data = curvature
    
   plt.imshow(slice_data, cmap='viridis')
   plt.colorbar(label=f'Curvature Slice at index {slice_idx}')
   plt.title(f'Curvature Tensor Slice for {layer_name}')
   
  if output_path:
   plt.savefig(output_path, bbox_inches='tight', dpi=300)
   self.logger.info(f"Visualization saved to {output_path}")
  else:
   plt.show()

 def get_curvature_statistics(self, layer_name: str) -> Dict[str, float]:
  """
  Get statistical summary of curvature for a layer.
  
  Args:
   layer_name: Name of the layer to analyze
   
  Returns:
   Dictionary containing curvature statistics
  """
  if layer_name not in self.curvature_tensors:
   raise KeyError(f"Layer {layer_name} not found in curvature tensors")
   
  curvature = self.curvature_tensors[layer_name]
  
  # Flatten the curvature tensor for statistics
  flat_curvature = curvature.flatten()
  
  stats = {
      'mean': float(np.mean(flat_curvature)),
      'std': float(np.std(flat_curvature)),
      'min': float(np.min(flat_curvature)),
      'max': float(np.max(flat_curvature)),
      'median': float(np.median(flat_curvature)),
      'shape': curvature.shape # Add shape for context
  }
  if self.SCIPY_STATS_AVAILABLE: # Use the instance flag
      stats['skewness'] = float(scipy_stats.skew(flat_curvature))
      stats['kurtosis'] = float(scipy_stats.kurtosis(flat_curvature))
  else:
      stats['skewness'] = 0.0 # Default if scipy.stats not available
      stats['kurtosis'] = 0.0
  return stats

 def compare_layers(self, layer1: str, layer2: str) -> Dict[str, Any]:
  """
  Compare curvature properties between two layers.
  
  Args:
   layer1: Name of first layer
   layer2: Name of second layer
   
  Returns:
   Dictionary containing comparison metrics
  """
  if layer1 not in self.curvature_tensors or layer2 not in self.curvature_tensors:
   raise KeyError("One or both layers not found in curvature tensors")
   
  stats1 = self.get_curvature_statistics(layer1)
  stats2 = self.get_curvature_statistics(layer2)
  
  return {
   'layer1': layer1,
   'layer2': layer2,
   'mean_difference': stats1['mean'] - stats2['mean'],
   'std_ratio': stats1['std'] / (stats2['std'] + self.eps),
   'range_ratio': (stats1['max'] - stats1['min']) / (stats2['max'] - stats2['min'] + self.eps),
   'correlation': self._compute_layer_correlation(layer1, layer2) if self._shapes_compatible(layer1, layer2) else None
  }

 def _shapes_compatible(self, layer1: str, layer2: str) -> bool:
  """Check if two layers have compatible shapes for correlation."""
  shape1 = self.curvature_tensors[layer1].shape
  shape2 = self.curvature_tensors[layer2].shape
  return shape1 == shape2

 def _compute_layer_correlation(self, layer1: str, layer2: str) -> float:
  """Compute correlation between curvature tensors of two layers."""
  tensor1 = self.curvature_tensors[layer1].flatten()
  tensor2 = self.curvature_tensors[layer2].flatten()
  
  correlation_matrix = np.corrcoef(tensor1, tensor2)
  return float(correlation_matrix[0, 1])

 def export_results(self, output_path: str, include_raw_data: bool = False) -> None:
  """
  Export analysis results to JSON file.
  
  Args:
   output_path: Path to save results
   include_raw_data: Whether to include raw curvature tensors
  """
  results = {
   'metadata': {
    'tensor_count': len(self.tensor_field),
    'metric_type': self.metric_type,
    'epsilon': self.eps,
    'analysis_timestamp': str(np.datetime64('now'))
   },
   'layers': {}
  }
  
  for layer_name in self.tensor_field.keys():
   layer_results = {
    'tensor_shape': self.tensor_field[layer_name].shape,
    'has_curvature': layer_name in self.curvature_tensors
   }
   
   if layer_name in self.curvature_tensors:
    layer_results['curvature_statistics'] = self.get_curvature_statistics(layer_name)
    
    if include_raw_data:
     # Convert to list for JSON serialization
     layer_results['curvature_tensor'] = self.curvature_tensors[layer_name].tolist()
   
   results['layers'][layer_name] = layer_results
  
  with open(output_path, 'w') as f:
   json.dump(results, f, indent=2, ensure_ascii=False)
  
  self.logger.info(f"Results exported to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="SVELTE Attention Topology CLI")
    parser.add_argument('model', type=str, help='Path to GGUF model file')
    parser.add_argument('--output', '-o', type=str, help='Output path for results')
    parser.add_argument('--method', '-m', type=str, default='riemann', 
                       choices=['riemann', 'ricci', 'scalar', 'sectional'],
                       help='Curvature computation method')
    parser.add_argument('--layer', '-l', type=str, help='Specific layer to analyze')
    args = parser.parse_args()
    
    try:
        # Import required modules
        from src.tensor_analysis.gguf_parser import GGUFParser
        from src.tensor_analysis.tensor_field import TensorFieldConstructor
        
        # Parse the GGUF model
        gguf = GGUFParser(args.model)
        gguf.parse()
        
        # Construct tensor field
        tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
        tensor_field_obj = tensor_field_constructor.construct()
        
        # Convert TensorField object to dictionary format expected by AttentionTopologySystem
        if hasattr(tensor_field_obj, 'tensors'):
            tensor_dict = tensor_field_obj.tensors
        elif isinstance(tensor_field_obj, dict):
            tensor_dict = tensor_field_obj
        else:
            # Try to extract tensor data from the object
            tensor_dict = {}
            for attr_name in dir(tensor_field_obj):
                if not attr_name.startswith('_'):
                    attr_value = getattr(tensor_field_obj, attr_name)
                    if isinstance(attr_value, np.ndarray):
                        tensor_dict[attr_name] = attr_value
        
        # Create attention topology system
        attention_topology = AttentionTopologySystem(tensor_dict)
        
        # Compute curvature
        method = CurvatureMethod(args.method)
        curvature = attention_topology.compute_curvature(method)
        
        # Analyze specific layer if requested
        if args.layer:
            if args.layer in tensor_dict:
                metrics = attention_topology.analyze_topology(args.layer)
                print(f"Analysis for layer {args.layer}:")
                print(f"  Entropy: {metrics.entropy:.4f}")
                print(f"  Curvature shape: {metrics.curvature.shape}")
                if metrics.eigenvalues is not None:
                    print(f"  Eigenvalues (top 5): {metrics.eigenvalues[-5:]}")
                print(f"  Homology: {metrics.homology}")
            else:
                print(f"Layer {args.layer} not found in tensor field")
                print(f"Available layers: {list(tensor_dict.keys())}")
        
        # Output results
        if args.output:
            output_data = {
                'curvature_method': args.method,
                'layers': list(tensor_dict.keys()),
                'curvature_shapes': {name: tensor.shape for name, tensor in curvature.items()},
                'analysis_summary': {
                    'total_layers': len(tensor_dict),
                    'method_used': args.method
                }
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}")
        else:
            # Print summary to console
            print(f"Curvature analysis complete using {args.method} method")
            print(f"Processed {len(tensor_dict)} layers:")
            for name, tensor in curvature.items():
                print(f"  {name}: {tensor.shape}")
                
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required modules are available")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
