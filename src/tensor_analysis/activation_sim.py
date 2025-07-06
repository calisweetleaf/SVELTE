"""
Activation Space Simulator for SVELTE Framework.
Simulates forward-pass activations and records patterns across neural network layers.
Provides comprehensive analysis tools for activation distributions and neuron behavior.

author: Morpheus
version: 1.0.0
date: 2025-05-01
description: This module simulates the forward pass of a neural network, records activations, and provides analysis tools for understanding activation distributions and neuron behavior. It supports multiple activation functions, batch processing, and detailed statistical analysis of neuron behaviors.
ID: 001
SHA-256: abcdef1234567890abcdef1234567890abcdef123456
"""

import numpy as np
import scipy.stats as stats
import scipy.ndimage as ndimage
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import os
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import h5py
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ActivationMetrics:
 """Data class for storing activation metrics."""
 mean: float
 median: float
 std_dev: float
 kurtosis: float
 skewness: float
 min_val: float
 max_val: float
 sparsity: float  # percentage of zero/near-zero values
 l1_norm: float
 l2_norm: float

class ActivationSpaceSimulator:
 """
 Simulates neural network forward-pass activations and provides
 comprehensive analysis of activation patterns and distributions.
 
 Supports multiple activation functions, batch processing, and
 detailed statistical analysis of neuron behaviors.
 """
 
 # Common activation functions
 ACTIVATION_FUNCTIONS = {
  'relu': lambda x: np.maximum(0, x),
  'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
  'tanh': lambda x: np.tanh(x),
  'gelu': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
  'swish': lambda x: x * (1 / (1 + np.exp(-x))),
  'identity': lambda x: x,
 }
 
 def __init__(self, tensor_field: Dict[str, np.ndarray], 
     activation_fn: Union[str, Dict[str, str]] = 'relu',
     epsilon: float = 1e-10,
     cache_dir: Optional[str] = None):
  """
  Initialize the activation space simulator.
  
  Args:
   tensor_field: Dictionary mapping layer names to weight tensors
   activation_fn: Either a single activation function name for all layers,
        or a dict mapping layer names to activation functions
   epsilon: Small value to prevent division by zero
   cache_dir: Directory to cache activation maps
  """
  self.tensor_field = tensor_field
  self.activations = {}
  self.activation_gradients = {}
  self.layer_metrics = {}
  self.epsilon = epsilon
  self.cache_dir = cache_dir
  
  # Set up activation functions for each layer
  self.layer_activation_fns = {}
  if isinstance(activation_fn, str):
   if activation_fn not in self.ACTIVATION_FUNCTIONS:
    raise ValueError(f"Unknown activation function: {activation_fn}")
   for layer_name in tensor_field:
    self.layer_activation_fns[layer_name] = self.ACTIVATION_FUNCTIONS[activation_fn]
  elif isinstance(activation_fn, dict):
   for layer_name, fn_name in activation_fn.items():
    if fn_name not in self.ACTIVATION_FUNCTIONS:
     raise ValueError(f"Unknown activation function: {fn_name}")
    self.layer_activation_fns[layer_name] = self.ACTIVATION_FUNCTIONS[fn_name]
  else:
   raise TypeError("activation_fn must be a string or dictionary")
  
  logger.info(f"Initialized ActivationSpaceSimulator with {len(tensor_field)} layers")
  
  # Create cache directory if specified
  if self.cache_dir:
   os.makedirs(self.cache_dir, exist_ok=True)

 def simulate(self, inputs: np.ndarray, batch_mode: bool = False) -> Dict[str, np.ndarray]:
  """
  Simulate forward pass and record activations for all layers.
  
  Args:
   inputs: Input tensor of shape (batch_size, feature_dim) or (feature_dim,)
   batch_mode: Whether to process as batch or single sample
   
  Returns:
   Dictionary mapping layer names to activation tensors
  """
  try:
   # Ensure input is properly shaped
   if not batch_mode and inputs.ndim == 1:
    inputs = inputs.reshape(1, -1)
   
   logger.info(f"Simulating forward pass with input shape: {inputs.shape}")
   self.activations = {}
   current_activation = inputs
   
   # Process each layer in order
   for i, (layer_name, weights) in enumerate(sorted(self.tensor_field.items())):
    if i == 0:  # First layer
     # Validate input dimensions
     if inputs.shape[-1] != weights.shape[0]:
      raise ValueError(f"Input dimension {inputs.shape[-1]} doesn't match first layer dimension {weights.shape[0]}")
    
    # Compute pre-activation
    if weights.ndim == 2:  # Dense layer
     pre_activation = np.matmul(current_activation, weights)
    elif weights.ndim == 4:  # Conv layer (simplified)
     # This is a simplified implementation - production code would need proper convolution
     logger.warning("Using simplified convolution implementation")
     batch_size = current_activation.shape[0]
     output_shape = (batch_size,) + weights.shape[0:1] + weights.shape[2:]
     pre_activation = np.zeros(output_shape)
     for b in range(batch_size):
      for i in range(weights.shape[0]):
       for j in range(weights.shape[1]):
        pre_activation[b, i] += np.convolve(current_activation[b, j], weights[i, j], mode='same')
    else:
     raise ValueError(f"Unsupported weight tensor shape: {weights.shape}")
    
    # Apply activation function
    activation_fn = self.layer_activation_fns.get(layer_name, 
               self.ACTIVATION_FUNCTIONS['identity'])
    current_activation = activation_fn(pre_activation)
    
    # Store activation
    self.activations[layer_name] = current_activation.copy()
    
    logger.debug(f"Layer {layer_name}: activation shape {current_activation.shape}")
   
   logger.info(f"Forward pass complete. Generated activations for {len(self.activations)} layers")
   return self.activations
   
  except Exception as e:
   logger.error(f"Error during simulation: {str(e)}")
   raise

 def analyze_distribution(self, layers: Optional[List[str]] = None) -> Dict[str, ActivationMetrics]:
  """
  Analyze activation distributions layer by layer.
  
  Args:
   layers: List of layer names to analyze. If None, analyze all layers.
   
  Returns:
   Dictionary mapping layer names to activation metrics
  """
  if not self.activations:
   raise ValueError("No activations available. Run simulate() first.")
   
  layers_to_analyze = layers if layers else list(self.activations.keys())
  self.layer_metrics = {}
  
  for layer_name in layers_to_analyze:
   if layer_name not in self.activations:
    logger.warning(f"Layer {layer_name} not found in activations, skipping...")
    continue
    
   activation = self.activations[layer_name]
   flat_activation = activation.reshape(-1)
   
   # Calculate metrics
   metrics = ActivationMetrics(
    mean=float(np.mean(flat_activation)),
    median=float(np.median(flat_activation)),
    std_dev=float(np.std(flat_activation)),
    kurtosis=float(stats.kurtosis(flat_activation)),
    skewness=float(stats.skew(flat_activation)),
    min_val=float(np.min(flat_activation)),
    max_val=float(np.max(flat_activation)),
    sparsity=float(np.mean(np.abs(flat_activation) < self.epsilon)),
    l1_norm=float(np.sum(np.abs(flat_activation))),
    l2_norm=float(np.sqrt(np.sum(np.square(flat_activation))))
   )
   
   self.layer_metrics[layer_name] = metrics
   
   logger.info(f"Layer {layer_name} metrics: mean={metrics.mean:.4f}, "
        f"std={metrics.std_dev:.4f}, sparsity={metrics.sparsity:.4f}")
   
  return self.layer_metrics

 def find_influential_neurons(self, layer_name: str, top_k: int = 10) -> List[Tuple[int, float]]:
  """
  Find the most influential neurons in a given layer based on activation magnitude.
  
  Args:
   layer_name: Name of the layer to analyze
   top_k: Number of top neurons to return
   
  Returns:
   List of tuples (neuron_idx, activation_value) for top neurons
  """
  if layer_name not in self.activations:
   raise ValueError(f"Layer {layer_name} not found in activations")
   
  activation = self.activations[layer_name]
  
  # For multi-dimensional activations, reshape to 2D (samples, neurons)
  if activation.ndim > 2:
   activation = activation.reshape(activation.shape[0], -1)
   
  # Average across samples if batched
  if activation.ndim == 2:
   avg_activation = np.mean(np.abs(activation), axis=0)
  else:
   avg_activation = np.abs(activation)
   
  # Get top-k indices and values
  top_indices = np.argsort(-avg_activation)[:top_k]
  top_values = avg_activation[top_indices]
  
  return [(int(idx), float(val)) for idx, val in zip(top_indices, top_values)]

 def compare_layers(self, metric: str = 'sparsity') -> Dict[str, float]:
  """
  Compare layers based on a specific metric.
  
  Args:
   metric: Metric to compare ('mean', 'std_dev', 'sparsity', etc.)
   
  Returns:
   Dictionary mapping layer names to metric values
  """
  if not self.layer_metrics:
   self.analyze_distribution()
   
  result = {}
  for layer_name, metrics in self.layer_metrics.items():
   if hasattr(metrics, metric):
    result[layer_name] = getattr(metrics, metric)
   else:
    raise ValueError(f"Unknown metric: {metric}")
    
  return result

 def save_activation_maps(self, filepath: Optional[str] = None) -> str:
  """
  Save activation maps to disk.
  
  Args:
   filepath: Path to save activations. If None, use cache_dir/timestamp.
   
  Returns:
   Path where activations were saved
  """
  if not self.activations:
   raise ValueError("No activations to save")
   
  if filepath is None:
   if self.cache_dir is None:
    raise ValueError("No filepath or cache_dir specified")
    
   import time
   timestamp = int(time.time())
   filepath = os.path.join(self.cache_dir, f"activations_{timestamp}.npz")
   
  # Convert to list of arrays for saving
  save_dict = {k: v for k, v in self.activations.items()}
  
  np.savez_compressed(filepath, **save_dict)
  logger.info(f"Saved activation maps to {filepath}")
  
  return filepath

 def load_activation_maps(self, filepath: str) -> Dict[str, np.ndarray]:
  """
  Load activation maps from disk.
  
  Args:
   filepath: Path to load activations from
   
  Returns:
   Dictionary of loaded activations
  """
  if not os.path.exists(filepath):
   raise FileNotFoundError(f"Activation file not found: {filepath}")
   
  loaded = np.load(filepath)
  self.activations = {k: loaded[k] for k in loaded.files}
  
  logger.info(f"Loaded activation maps for {len(self.activations)} layers from {filepath}")
  return self.activations

 def visualize_distribution(self, layer_name: str, save_path: Optional[str] = None):
  """
  Visualize activation distribution for a specific layer.
  
  Args:
   layer_name: Name of the layer to visualize
   save_path: Path to save the visualization. If None, display the plot.
  """
  if layer_name not in self.activations:
   raise ValueError(f"Layer {layer_name} not found in activations")
   
  activation = self.activations[layer_name]
  flat_activation = activation.reshape(-1)
  
  # Create comprehensive visualization
  fig, axes = plt.subplots(2, 2, figsize=(15, 12))
  fig.suptitle(f'Activation Distribution Analysis - Layer: {layer_name}', fontsize=16)
  
  # Histogram
  axes[0, 0].hist(flat_activation, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
  axes[0, 0].set_title('Activation Histogram')
  axes[0, 0].set_xlabel('Activation Value')
  axes[0, 0].set_ylabel('Density')
  axes[0, 0].grid(True, alpha=0.3)
  
  # Box plot
  axes[0, 1].boxplot(flat_activation, vert=True, patch_artist=True, 
                     boxprops=dict(facecolor='lightcoral', alpha=0.7))
  axes[0, 1].set_title('Box Plot')
  axes[0, 1].set_ylabel('Activation Value')
  axes[0, 1].grid(True, alpha=0.3)
  
  # Q-Q plot
  stats.probplot(flat_activation, dist="norm", plot=axes[1, 0])
  axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
  axes[1, 0].grid(True, alpha=0.3)
  
  # Cumulative distribution
  sorted_values = np.sort(flat_activation)
  cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
  axes[1, 1].plot(sorted_values, cumulative, 'b-', linewidth=2)
  axes[1, 1].set_title('Cumulative Distribution')
  axes[1, 1].set_xlabel('Activation Value')
  axes[1, 1].set_ylabel('Cumulative Probability')
  axes[1, 1].grid(True, alpha=0.3)
  
  # Add statistics text
  metrics = self.layer_metrics.get(layer_name)
  if metrics:
   stats_text = f'Mean: {metrics.mean:.4f}\nStd: {metrics.std_dev:.4f}\n'
   stats_text += f'Sparsity: {metrics.sparsity:.4f}\nSkewness: {metrics.skewness:.4f}\n'
   stats_text += f'Kurtosis: {metrics.kurtosis:.4f}'
   fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
  
  plt.tight_layout()
  
  if save_path:
   plt.savefig(save_path, dpi=300, bbox_inches='tight')
   logger.info(f"Saved visualization to {save_path}")
  else:
   plt.show()
  
  plt.close()

 def compute_activation_gradients(self, layer_name: str, method: str = 'sobel') -> Dict[str, np.ndarray]:
  """
  Compute gradients of activation maps for pattern analysis.
  
  Args:
   layer_name: Name of the layer to analyze
   method: Gradient computation method ('sobel', 'scharr', 'prewitt', 'central_diff')
   
  Returns:
   Dictionary containing gradient magnitude and direction maps
  """
  if layer_name not in self.activations:
   raise ValueError(f"Layer {layer_name} not found in activations")
   
  activation = self.activations[layer_name]
  
  # For multi-dimensional activations, compute gradients for each 2D slice
  if activation.ndim == 2:
   activation_2d = activation
  elif activation.ndim == 3:
   # Average across channels or take first channel
   activation_2d = np.mean(activation, axis=0) if activation.shape[0] > 1 else activation[0]
  else:
   # Reshape to 2D for gradient computation
   activation_2d = activation.reshape(activation.shape[0], -1)
  
  if method == 'sobel':
   grad_x = ndimage.sobel(activation_2d, axis=1)
   grad_y = ndimage.sobel(activation_2d, axis=0)
  elif method == 'scharr':
   grad_x = ndimage.prewitt(activation_2d, axis=1)  # Scipy doesn't have Scharr, use Prewitt
   grad_y = ndimage.prewitt(activation_2d, axis=0)
  elif method == 'prewitt':
   grad_x = ndimage.prewitt(activation_2d, axis=1)
   grad_y = ndimage.prewitt(activation_2d, axis=0)
  elif method == 'central_diff':
   grad_x = np.gradient(activation_2d, axis=1)
   grad_y = np.gradient(activation_2d, axis=0)
  else:
   raise ValueError(f"Unknown gradient method: {method}")
  
  # Compute magnitude and direction
  grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
  grad_direction = np.arctan2(grad_y, grad_x)
  
  gradients = {
   'magnitude': grad_magnitude,
   'direction': grad_direction,
   'grad_x': grad_x,
   'grad_y': grad_y
  }
  
  # Store in instance variable
  if layer_name not in self.activation_gradients:
   self.activation_gradients[layer_name] = {}
  self.activation_gradients[layer_name][method] = gradients
  
  return gradients

 def detect_activation_patterns(self, layer_name: str, pattern_type: str = 'peaks') -> Dict[str, Any]:
  """
  Detect specific patterns in activation maps.
  
  Args:
   layer_name: Name of the layer to analyze
   pattern_type: Type of pattern to detect ('peaks', 'valleys', 'edges', 'blobs')
   
  Returns:
   Dictionary containing detected patterns and their properties
  """
  if layer_name not in self.activations:
   raise ValueError(f"Layer {layer_name} not found in activations")
   
  activation = self.activations[layer_name]
  flat_activation = activation.reshape(-1)
  
  patterns = {}
  
  if pattern_type == 'peaks':
   # Find local maxima
   threshold = np.percentile(flat_activation, 95)
   peaks = flat_activation > threshold
   patterns['peak_indices'] = np.where(peaks)[0]
   patterns['peak_values'] = flat_activation[peaks]
   patterns['peak_count'] = len(patterns['peak_indices'])
   
  elif pattern_type == 'valleys':
   # Find local minima
   threshold = np.percentile(flat_activation, 5)
   valleys = flat_activation < threshold
   patterns['valley_indices'] = np.where(valleys)[0]
   patterns['valley_values'] = flat_activation[valleys]
   patterns['valley_count'] = len(patterns['valley_indices'])
   
  elif pattern_type == 'edges':
   # Use gradient magnitude for edge detection
   if layer_name not in self.activation_gradients:
    self.compute_activation_gradients(layer_name)
   
   grad_mag = self.activation_gradients[layer_name]['sobel']['magnitude']
   edge_threshold = np.percentile(grad_mag.flatten(), 90)
   edges = grad_mag > edge_threshold
   patterns['edge_map'] = edges
   patterns['edge_strength'] = grad_mag[edges]
   patterns['edge_count'] = np.sum(edges)
   
  elif pattern_type == 'blobs':
   # Simple blob detection using local variance
   if activation.ndim >= 2:
    # Use a sliding window approach for blob detection
    window_size = min(5, activation.shape[-1] // 10)
    blob_map = ndimage.uniform_filter(activation, size=window_size)
    blob_variance = ndimage.uniform_filter(activation**2, size=window_size) - blob_map**2
    
    blob_threshold = np.percentile(blob_variance.flatten(), 95)
    blobs = blob_variance > blob_threshold
    patterns['blob_map'] = blobs
    patterns['blob_variance'] = blob_variance[blobs]
    patterns['blob_count'] = np.sum(blobs)
   else:
    patterns['blob_count'] = 0
    patterns['blob_map'] = np.zeros_like(activation, dtype=bool)
  
  return patterns

 def cluster_neuron_activations(self, layer_name: str, n_clusters: int = 5, 
                        method: str = 'kmeans', reduce_dim: bool = True) -> Dict[str, Any]:
  """
  Cluster neuron activations to identify functional groups.
  
  Args:
   layer_name: Name of the layer to analyze
   n_clusters: Number of clusters to create
   method: Clustering method ('kmeans', 'hierarchical')
   reduce_dim: Whether to reduce dimensionality before clustering
   
  Returns:
   Dictionary containing cluster assignments and properties
  """
  if layer_name not in self.activations:
   raise ValueError(f"Layer {layer_name} not found in activations")
   
  activation = self.activations[layer_name]
  
  # Reshape for clustering (samples x features)
  if activation.ndim == 1:
   data = activation.reshape(-1, 1)
  else:
   data = activation.reshape(activation.shape[0], -1)
  
  # Dimensionality reduction if requested
  if reduce_dim and data.shape[1] > 50:
   pca = PCA(n_components=min(50, data.shape[1]))
   data = pca.fit_transform(data)
   explained_variance = pca.explained_variance_ratio_
  else:
   explained_variance = None
  
  # Perform clustering
  if method == 'kmeans':
   clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
   cluster_labels = clusterer.fit_predict(data)
   cluster_centers = clusterer.cluster_centers_
   inertia = clusterer.inertia_
  else:
   raise ValueError(f"Clustering method '{method}' not supported")
  
  # Analyze clusters
  cluster_info = {}
  for i in range(n_clusters):
   cluster_mask = cluster_labels == i
   cluster_data = data[cluster_mask]
   
   cluster_info[f'cluster_{i}'] = {
    'size': np.sum(cluster_mask),
    'mean': np.mean(cluster_data, axis=0),
    'std': np.std(cluster_data, axis=0),
    'indices': np.where(cluster_mask)[0]
   }
  
  results = {
   'cluster_labels': cluster_labels,
   'cluster_centers': cluster_centers,
   'cluster_info': cluster_info,
   'n_clusters': n_clusters,
   'inertia': inertia if method == 'kmeans' else None,
   'explained_variance': explained_variance,
   'method': method
  }
  
  return results

 def compute_activation_statistics(self, layer_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
  """
  Compute comprehensive statistics for activation patterns.
  
  Args:
   layer_names: List of layer names to analyze. If None, analyze all layers.
   
  Returns:
   Dictionary containing detailed statistics for each layer
  """
  if not self.activations:
   raise ValueError("No activations available. Run simulate() first.")
   
  layers_to_analyze = layer_names if layer_names else list(self.activations.keys())
  statistics = {}
  
  for layer_name in layers_to_analyze:
   if layer_name not in self.activations:
    logger.warning(f"Layer {layer_name} not found in activations, skipping...")
    continue
    
   activation = self.activations[layer_name]
   flat_activation = activation.reshape(-1)
   
   # Basic statistics
   stats_dict = {
    'shape': activation.shape,
    'total_elements': activation.size,
    'mean': float(np.mean(flat_activation)),
    'median': float(np.median(flat_activation)),
    'std': float(np.std(flat_activation)),
    'var': float(np.var(flat_activation)),
    'min': float(np.min(flat_activation)),
    'max': float(np.max(flat_activation)),
    'range': float(np.ptp(flat_activation)),
    'sum': float(np.sum(flat_activation)),
    'abs_sum': float(np.sum(np.abs(flat_activation))),
    'squared_sum': float(np.sum(flat_activation**2)),
   }
   
   # Percentiles
   percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
   for p in percentiles:
    stats_dict[f'percentile_{p}'] = float(np.percentile(flat_activation, p))
   
   # Distribution characteristics
   stats_dict['skewness'] = float(stats.skew(flat_activation))
   stats_dict['kurtosis'] = float(stats.kurtosis(flat_activation))
   stats_dict['sparsity'] = float(np.mean(np.abs(flat_activation) < self.epsilon))
   
   # Norms
   stats_dict['l1_norm'] = float(np.sum(np.abs(flat_activation)))
   stats_dict['l2_norm'] = float(np.sqrt(np.sum(flat_activation**2)))
   stats_dict['l_inf_norm'] = float(np.max(np.abs(flat_activation)))
   
   # Information theoretic measures
   hist, _ = np.histogram(flat_activation, bins=50, density=True)
   hist = hist[hist > 0]  # Remove zero bins
   if len(hist) > 0:
    stats_dict['entropy'] = float(-np.sum(hist * np.log2(hist + self.epsilon)))
   else:
    stats_dict['entropy'] = 0.0
   
   # Zero/negative/positive counts
   stats_dict['zero_count'] = int(np.sum(np.abs(flat_activation) < self.epsilon))
   stats_dict['negative_count'] = int(np.sum(flat_activation < 0))
   stats_dict['positive_count'] = int(np.sum(flat_activation > 0))
   
   # Activation density (non-zero elements)
   stats_dict['activation_density'] = 1.0 - stats_dict['sparsity']
   
   statistics[layer_name] = stats_dict
  
  return statistics

 def generate_activation_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
  """
  Generate a comprehensive activation analysis report.
  
  Args:
   output_path: Path to save the report. If None, return the report dictionary.
   
  Returns:
   Dictionary containing the complete analysis report
  """
  if not self.activations:
   raise ValueError("No activations available. Run simulate() first.")
   
  # Ensure we have metrics calculated
  if not self.layer_metrics:
   self.analyze_distribution()
   
  # Generate comprehensive statistics
  stats = self.compute_activation_statistics()
  
  # Create report
  report = {
   'metadata': {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'total_layers': len(self.activations),
    'total_parameters': sum(act.size for act in self.activations.values()),
    'activation_functions': {name: str(func) for name, func in self.layer_activation_fns.items()},
    'epsilon': self.epsilon
   },
   'layer_statistics': stats,
   'layer_metrics': {name: {
    'mean': metrics.mean,
    'std_dev': metrics.std_dev,
    'sparsity': metrics.sparsity,
    'skewness': metrics.skewness,
    'kurtosis': metrics.kurtosis,
    'l1_norm': metrics.l1_norm,
    'l2_norm': metrics.l2_norm
   } for name, metrics in self.layer_metrics.items()},
   'summary': {
    'most_sparse_layer': min(self.layer_metrics.keys(), 
                 key=lambda x: self.layer_metrics[x].sparsity),
    'least_sparse_layer': max(self.layer_metrics.keys(), 
                 key=lambda x: self.layer_metrics[x].sparsity),
    'highest_mean_activation': max(self.layer_metrics.keys(), 
                    key=lambda x: self.layer_metrics[x].mean),
    'lowest_mean_activation': min(self.layer_metrics.keys(), 
                   key=lambda x: self.layer_metrics[x].mean),
    'most_variable_layer': max(self.layer_metrics.keys(), 
                  key=lambda x: self.layer_metrics[x].std_dev),
    'least_variable_layer': min(self.layer_metrics.keys(), 
                   key=lambda x: self.layer_metrics[x].std_dev)
   }
  }
  
  if output_path:
   # Save as JSON
   with open(output_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)
   logger.info(f"Activation report saved to {output_path}")
  
  return report

 def export_activations(self, output_path: str, format: str = 'hdf5', 
                       include_metadata: bool = True) -> None:
  """
  Export activation maps to various formats.
  
  Args:
   output_path: Path to save the exported data
   format: Export format ('hdf5', 'npz', 'pickle', 'json')
   include_metadata: Whether to include metadata and metrics
  """
  if not self.activations:
   raise ValueError("No activations available. Run simulate() first.")
   
  export_data = {
   'activations': self.activations,
   'layer_activation_fns': {name: str(func) for name, func in self.layer_activation_fns.items()},
   'epsilon': self.epsilon
  }
  
  if include_metadata:
   export_data['layer_metrics'] = self.layer_metrics
   export_data['activation_gradients'] = self.activation_gradients
   export_data['metadata'] = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'total_layers': len(self.activations),
    'total_parameters': sum(act.size for act in self.activations.values())
   }
  
  if format == 'hdf5':
   with h5py.File(output_path, 'w') as f:
    # Save activations
    act_group = f.create_group('activations')
    for name, activation in self.activations.items():
     act_group.create_dataset(name, data=activation, compression='gzip')
    
    # Save metadata
    if include_metadata:
     meta_group = f.create_group('metadata')
     for key, value in export_data['metadata'].items():
      meta_group.attrs[key] = value
      
  elif format == 'npz':
   save_dict = {}
   for name, activation in self.activations.items():
    save_dict[f'activation_{name}'] = activation
   
   if include_metadata:
    save_dict['metadata'] = json.dumps(export_data['metadata'])
   
   np.savez_compressed(output_path, **save_dict)
   
  elif format == 'pickle':
   with open(output_path, 'wb') as f:
    pickle.dump(export_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
  elif format == 'json':
   # Convert numpy arrays to lists for JSON serialization
   json_data = {}
   for name, activation in self.activations.items():
    json_data[f'activation_{name}'] = activation.tolist()
   
   if include_metadata:
    json_data['metadata'] = export_data['metadata']
   
   with open(output_path, 'w') as f:
    json.dump(json_data, f, indent=2)
    
  else:
   raise ValueError(f"Unsupported export format: {format}")
  
  logger.info(f"Exported activations to {output_path} in {format} format")

 def batch_simulate(self, input_batch: List[np.ndarray], 
                   parallel: bool = True, max_workers: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
  """
  Simulate forward pass for multiple inputs in batch.
  
  Args:
   input_batch: List of input arrays
   parallel: Whether to use parallel processing
   max_workers: Maximum number of parallel workers
   
  Returns:
   List of activation dictionaries for each input
  """
  if not input_batch:
   raise ValueError("Input batch is empty")
   
  if not parallel or len(input_batch) == 1:
   # Sequential processing
   results = []
   for i, inputs in enumerate(input_batch):
    logger.info(f"Processing input {i+1}/{len(input_batch)}")
    result = self.simulate(inputs, batch_mode=True)
    results.append(result)
   return results
  
  # Parallel processing
  if max_workers is None:
   max_workers = min(len(input_batch), os.cpu_count() or 4)
  results: List[Dict[str, np.ndarray]] = [{}] * len(input_batch)
  
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
   # Submit all tasks
   future_to_index = {
    executor.submit(self.simulate, inputs, True): i 
    for i, inputs in enumerate(input_batch)
   }
   
   # Collect results
   for future in as_completed(future_to_index):
    index = future_to_index[future]
    try:
     results[index] = future.result()
     logger.info(f"Completed processing input {index+1}/{len(input_batch)}")
    except Exception as e:
     logger.error(f"Error processing input {index+1}: {str(e)}")
     results[index] = {}
  
  return results

 def visualize_activation_heatmap(self, layer_names: Optional[List[str]] = None, 
                                 save_path: Optional[str] = None) -> None:
  """
  Create heatmap visualization of activation patterns across layers.
  
  Args:
   layer_names: List of layer names to include. If None, include all layers.
   save_path: Path to save the visualization
  """
  if not self.activations:
   raise ValueError("No activations available. Run simulate() first.")
   
  layers_to_plot = layer_names if layer_names else list(self.activations.keys())
  
  # Create activation summary matrix
  activation_matrix = []
  layer_labels = []
  
  for layer_name in layers_to_plot:
   if layer_name in self.activations:
    activation = self.activations[layer_name]
    # Use mean activation across all dimensions
    mean_activation = np.mean(activation.reshape(-1))
    activation_matrix.append([mean_activation])
    layer_labels.append(layer_name)
  
  activation_matrix = np.array(activation_matrix)
  
  # Create heatmap
  plt.figure(figsize=(12, max(8, len(layer_labels) * 0.5)))
  
  # Use seaborn for better heatmap
  sns.heatmap(activation_matrix, 
             yticklabels=layer_labels,
             xticklabels=['Mean Activation'],
             annot=True, 
             fmt='.4f', 
             cmap='RdYlBu_r',
             center=0,
             cbar_kws={'label': 'Activation Value'})
  
  plt.title('Activation Heatmap Across Layers')
  plt.xlabel('Activation Statistics')
  plt.ylabel('Layer Names')
  plt.tight_layout()
  
  if save_path:
   plt.savefig(save_path, dpi=300, bbox_inches='tight')
   logger.info(f"Saved activation heatmap to {save_path}")
  else:
   plt.show()
  
  plt.close()

 def get_layer_summary(self, layer_name: str) -> Dict[str, Any]:
  """
  Get comprehensive summary for a specific layer.
  
  Args:
   layer_name: Name of the layer to summarize
   
  Returns:
   Dictionary containing layer summary
  """
  if layer_name not in self.activations:
   raise ValueError(f"Layer {layer_name} not found in activations")
   
  activation = self.activations[layer_name]
  
  # Basic info
  summary = {
   'layer_name': layer_name,
   'shape': activation.shape,
   'total_parameters': activation.size,
   'activation_function': str(self.layer_activation_fns.get(layer_name, 'unknown')),
   'data_type': str(activation.dtype)
  }
  
  # Add metrics if available
  if layer_name in self.layer_metrics:
   metrics = self.layer_metrics[layer_name]
   summary.update({
    'mean_activation': metrics.mean,
    'std_activation': metrics.std_dev,
    'sparsity': metrics.sparsity,
    'skewness': metrics.skewness,
    'kurtosis': metrics.kurtosis,
    'l1_norm': metrics.l1_norm,
    'l2_norm': metrics.l2_norm
   })
  
  # Add gradient info if available
  if layer_name in self.activation_gradients:
   summary['gradient_methods'] = list(self.activation_gradients[layer_name].keys())
  
  return summary

 def reset(self) -> None:
  """Reset all stored activations and computed metrics."""
  self.activations.clear()
  self.activation_gradients.clear()
  self.layer_metrics.clear()
  logger.info("Reset activation simulator - cleared all stored data")

 def pca_analysis(self, layers: Optional[List[str]] = None, n_components: int = 2) -> Dict[str, np.ndarray]:
  """
  Perform PCA analysis on activations to reduce dimensionality.
  
  Args:
   layers: List of layer names to analyze. If None, analyze all layers.
   n_components: Number of PCA components to retain
   
  Returns:
   Dictionary mapping layer names to PCA results
  """
  from sklearn.decomposition import PCA
  
  if not self.activations:
   raise ValueError("No activations available. Run simulate() first.")
   
  layers_to_analyze = layers if layers else list(self.activations.keys())
  pca_results = {}
  
  for layer_name in layers_to_analyze:
   if layer_name not in self.activations:
    logger.warning(f"Layer {layer_name} not found in activations, skipping...")
    continue
    
   activation = self.activations[layer_name]
   
   # Reshape for PCA (flatten spatial dimensions)
   if activation.ndim > 2:
    original_shape = activation.shape
    n_samples = original_shape[0]
    n_features = np.prod(original_shape[1:])
    activation = activation.reshape(n_samples, n_features)
    
   # Perform PCA
   pca = PCA(n_components=n_components)
   pca_result = pca.fit_transform(activation)
   
   pca_results[layer_name] = pca_result
   logger.info(f"PCA for {layer_name}: retained {n_components} components")
   
  return pca_results

 def tsne_analysis(self, layers: Optional[List[str]] = None, n_components: int = 2, perplexity: int = 30) -> Dict[str, np.ndarray]:
  """
  Perform t-SNE analysis on activations for visualization.
  
  Args:
   layers: List of layer names to analyze. If None, analyze all layers.
   n_components: Number of t-SNE components to retain
   perplexity: Perplexity parameter for t-SNE
   
  Returns:
   Dictionary mapping layer names to t-SNE results
  """
  from sklearn.manifold import TSNE
  
  if not self.activations:
   raise ValueError("No activations available. Run simulate() first.")
   
  layers_to_analyze = layers if layers else list(self.activations.keys())
  tsne_results = {}
  
  for layer_name in layers_to_analyze:
   if layer_name not in self.activations:
    logger.warning(f"Layer {layer_name} not found in activations, skipping...")
    continue
    
   activation = self.activations[layer_name]
   
   # Reshape for t-SNE (flatten spatial dimensions)
   if activation.ndim > 2:
    original_shape = activation.shape
    n_samples = original_shape[0]
    n_features = np.prod(original_shape[1:])
    activation = activation.reshape(n_samples, n_features)
    
   # Perform t-SNE
   tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=0)
   tsne_result = tsne.fit_transform(activation)
   
   tsne_results[layer_name] = tsne_result
   logger.info(f"t-SNE for {layer_name}: retained {n_components} components, perplexity={perplexity}")
   
  return tsne_results

 def cluster_activations(self, layers: Optional[List[str]] = None, n_clusters: int = 3) -> Dict[str, np.ndarray]:
  """
  Cluster activations using KMeans clustering.
  
  Args:
   layers: List of layer names to analyze. If None, analyze all layers.
   n_clusters: Number of clusters for KMeans
   
  Returns:
   Dictionary mapping layer names to cluster labels
  """
  from sklearn.cluster import KMeans
  
  if not self.activations:
   raise ValueError("No activations available. Run simulate() first.")
   
  layers_to_analyze = layers if layers else list(self.activations.keys())
  cluster_results = {}
  
  for layer_name in layers_to_analyze:
   if layer_name not in self.activations:
    logger.warning(f"Layer {layer_name} not found in activations, skipping...")
    continue
    
   activation = self.activations[layer_name]
   
   # Reshape for clustering (flatten spatial dimensions)
   if activation.ndim > 2:
    original_shape = activation.shape
    n_samples = original_shape[0]
    n_features = np.prod(original_shape[1:])
    activation = activation.reshape(n_samples, n_features)
    
   # Perform KMeans clustering
   kmeans = KMeans(n_clusters=n_clusters, random_state=0)
   cluster_labels = kmeans.fit_predict(activation)
   
   cluster_results[layer_name] = cluster_labels
   logger.info(f"Clustering for {layer_name}: {n_clusters} clusters")
   
  return cluster_results

 def compute_activation_similarity(self, layer_names: Optional[List[str]] = None, 
                                 method: str = 'cosine') -> Dict[str, Dict[str, float]]:
  """
  Compute similarity between activation patterns across layers.
  
  Args:
   layer_names: List of layer names to compare. If None, compare all layers.
   method: Similarity method ('cosine', 'euclidean', 'correlation')
   
  Returns:
   Dictionary mapping layer pairs to similarity scores
  """
  if not self.activations:
   raise ValueError("No activations available. Run simulate() first.")
   
  layers_to_compare = layer_names if layer_names else list(self.activations.keys())
  similarity_matrix = {}
  
  for i, layer1 in enumerate(layers_to_compare):
   if layer1 not in self.activations:
    continue
    
   similarity_matrix[layer1] = {}
   activation1 = self.activations[layer1].flatten()
   
   for j, layer2 in enumerate(layers_to_compare):
    if layer2 not in self.activations:
     continue
     
    activation2 = self.activations[layer2].flatten()
    
    # Ensure same length by padding/truncating
    min_len = min(len(activation1), len(activation2))
    act1 = activation1[:min_len]
    act2 = activation2[:min_len]
    
    if method == 'cosine':
     # Cosine similarity
     dot_product = np.dot(act1, act2)
     norm1 = np.linalg.norm(act1)
     norm2 = np.linalg.norm(act2)
     similarity = dot_product / (norm1 * norm2 + self.epsilon)
    elif method == 'euclidean':
     # Euclidean distance (converted to similarity)
     distance = np.linalg.norm(act1 - act2)
     similarity = 1.0 / (1.0 + distance)
    elif method == 'correlation':
     # Pearson correlation
     if len(act1) > 1:
      correlation = np.corrcoef(act1, act2)[0, 1]
      similarity = correlation if not np.isnan(correlation) else 0.0
     else:
      similarity = 0.0
    else:
     raise ValueError(f"Unknown similarity method: {method}")
    
    similarity_matrix[layer1][layer2] = float(similarity)
  
  return similarity_matrix

 def detect_activation_anomalies(self, layer_names: Optional[List[str]] = None, 
                                method: str = 'isolation_forest', 
                                contamination: float = 0.1) -> Dict[str, Dict[str, Any]]:
  """
  Detect anomalous activation patterns in layers.
  
  Args:
   layer_names: List of layer names to analyze. If None, analyze all layers.
   method: Anomaly detection method ('isolation_forest', 'one_class_svm', 'local_outlier_factor')
   contamination: Expected proportion of anomalies
   
  Returns:
   Dictionary mapping layer names to anomaly detection results
  """
  if not self.activations:
   raise ValueError("No activations available. Run simulate() first.")
   
  layers_to_analyze = layer_names if layer_names else list(self.activations.keys())
  anomaly_results = {}
  
  for layer_name in layers_to_analyze:
   if layer_name not in self.activations:
    continue
    
   activation = self.activations[layer_name]
   
   # Reshape for anomaly detection
   if activation.ndim > 2:
    n_samples = activation.shape[0]
    n_features = np.prod(activation.shape[1:])
    data = activation.reshape(n_samples, n_features)
   else:
    data = activation
   
   # Apply anomaly detection
   if method == 'isolation_forest':
    from sklearn.ensemble import IsolationForest
    detector = IsolationForest(contamination=contamination, random_state=42)
   elif method == 'one_class_svm':
    from sklearn.svm import OneClassSVM
    detector = OneClassSVM(gamma='scale', nu=contamination)
   elif method == 'local_outlier_factor':
    from sklearn.neighbors import LocalOutlierFactor
    detector = LocalOutlierFactor(contamination=contamination, novelty=True)
   else:
    raise ValueError(f"Unknown anomaly detection method: {method}")
   
   # Fit and predict
   if method == 'local_outlier_factor':
    try:
     detector.fit(data)
     anomaly_scores = detector.negative_outlier_factor_
     anomaly_labels = detector.fit_predict(data)
    except:
     # Fallback if attribute doesn't exist
     anomaly_labels = detector.fit_predict(data)
     anomaly_scores = np.ones(len(data))
   else:
    anomaly_labels = detector.fit_predict(data)
    if hasattr(detector, 'score_samples'):
     anomaly_scores = detector.score_samples(data)
    elif hasattr(detector, 'decision_function'):
     anomaly_scores = detector.decision_function(data)
    else:
     anomaly_scores = np.ones(len(data))
   
   # Store results
   anomaly_results[layer_name] = {
    'anomaly_labels': anomaly_labels,
    'anomaly_scores': anomaly_scores,
    'anomaly_count': int(np.sum(anomaly_labels == -1)),
    'anomaly_ratio': float(np.mean(anomaly_labels == -1)),
    'method': method
   }
   
   logger.info(f"Anomaly detection for {layer_name}: {anomaly_results[layer_name]['anomaly_count']} anomalies detected")
  
  return anomaly_results

 def compute_layer_importance(self, method: str = 'gradient_norm') -> Dict[str, float]:
  """
  Compute importance scores for each layer.
  
  Args:
   method: Importance computation method ('gradient_norm', 'activation_magnitude', 'variance')
   
  Returns:
   Dictionary mapping layer names to importance scores
  """
  if not self.activations:
   raise ValueError("No activations available. Run simulate() first.")
   
  importance_scores = {}
  
  for layer_name, activation in self.activations.items():
   if method == 'gradient_norm':
    # Use gradient magnitude as importance
    if layer_name in self.activation_gradients:
     grad_info = list(self.activation_gradients[layer_name].values())[0]
     importance = float(np.mean(grad_info['magnitude']))
    else:
     # Compute gradient if not available
     grad_info = self.compute_activation_gradients(layer_name)
     importance = float(np.mean(grad_info['magnitude']))
     
   elif method == 'activation_magnitude':
    # Use mean absolute activation as importance
    importance = float(np.mean(np.abs(activation)))
    
   elif method == 'variance':
    # Use activation variance as importance
    importance = float(np.var(activation))
    
   else:
    raise ValueError(f"Unknown importance method: {method}")
   
   importance_scores[layer_name] = importance
  
  return importance_scores

 def visualize_layer_comparison(self, layer_names: List[str], 
                               metric: str = 'mean', 
                               save_path: Optional[str] = None) -> None:
  """
  Create visualization comparing layers across different metrics.
  
  Args:
   layer_names: List of layer names to compare
   metric: Metric to compare ('mean', 'std', 'sparsity', 'skewness', 'kurtosis')
   save_path: Path to save the visualization
  """
  if not self.layer_metrics:
   self.analyze_distribution()
   
  # Extract metric values
  values = []
  valid_layers = []
  
  for layer_name in layer_names:
   if layer_name in self.layer_metrics:
    metrics = self.layer_metrics[layer_name]
    if hasattr(metrics, metric):
     values.append(getattr(metrics, metric))
     valid_layers.append(layer_name)
  
  if not values:
   raise ValueError(f"No valid layers found for metric '{metric}'")
  
  # Create visualization
  plt.figure(figsize=(12, 8))
  
  # Bar plot
  bars = plt.bar(valid_layers, values, color='skyblue', edgecolor='navy', alpha=0.7)
  plt.xlabel('Layer Names')
  plt.ylabel(f'{metric.replace("_", " ").title()}')
  plt.title(f'Layer Comparison: {metric.replace("_", " ").title()}')
  plt.xticks(rotation=45, ha='right')
  plt.grid(True, alpha=0.3)
  
  # Add value labels on bars
  for bar, value in zip(bars, values):
   plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.01, 
       f'{value:.4f}', ha='center', va='bottom', fontsize=9)
  
  plt.tight_layout()
  
  if save_path:
   plt.savefig(save_path, dpi=300, bbox_inches='tight')
   logger.info(f"Saved layer comparison to {save_path}")
  else:
   plt.show()
  
  plt.close()

 def compute_neuron_selectivity(self, layer_name: str, 
                               method: str = 'max_response') -> Dict[str, Any]:
  """
  Compute selectivity measures for individual neurons.
  
  Args:
   layer_name: Name of the layer to analyze
   method: Selectivity method ('max_response', 'sparsity', 'lifetime_sparsity')
   
  Returns:
   Dictionary containing selectivity measures
  """
  if layer_name not in self.activations:
   raise ValueError(f"Layer {layer_name} not found in activations")
   
  activation = self.activations[layer_name]
  
  # Reshape to (samples, neurons)
  if activation.ndim > 2:
   n_samples = activation.shape[0]
   n_neurons = np.prod(activation.shape[1:])
   activation = activation.reshape(n_samples, n_neurons)
  
  selectivity_measures = {}
  
  if method == 'max_response':
   # Maximum response selectivity
   max_responses = np.max(activation, axis=0)
   mean_responses = np.mean(activation, axis=0)
   selectivity = max_responses / (mean_responses + self.epsilon)
   selectivity_measures['max_response_selectivity'] = selectivity
   
  elif method == 'sparsity':
   # Sparsity-based selectivity
   sparsity = np.mean(np.abs(activation) < self.epsilon, axis=0)
   selectivity_measures['sparsity_selectivity'] = sparsity
   
  elif method == 'lifetime_sparsity':
   # Lifetime sparsity measure
   lifetime_sparsity = []
   for neuron_idx in range(activation.shape[1]):
    neuron_activations = activation[:, neuron_idx]
    mean_act = np.mean(neuron_activations)
    mean_act_squared = np.mean(neuron_activations**2)
    if mean_act_squared > self.epsilon:
     ls = (mean_act**2) / mean_act_squared
    else:
     ls = 0.0
    lifetime_sparsity.append(ls)
   selectivity_measures['lifetime_sparsity'] = np.array(lifetime_sparsity)
   
  else:
   raise ValueError(f"Unknown selectivity method: {method}")
  
  # Add summary statistics
  for measure_name, measure_values in selectivity_measures.items():
   selectivity_measures[f'{measure_name}_mean'] = float(np.mean(measure_values))
   selectivity_measures[f'{measure_name}_std'] = float(np.std(measure_values))
   selectivity_measures[f'{measure_name}_median'] = float(np.median(measure_values))
  
  return selectivity_measures

 def advanced_clustering_analysis(self, layer_name: str, 
                                 methods: List[str] = ['kmeans', 'dbscan', 'hierarchical']) -> Dict[str, Any]:
  """
  Perform advanced clustering analysis with multiple methods.
  
  Args:
   layer_name: Name of the layer to analyze
   methods: List of clustering methods to apply
   
  Returns:
   Dictionary containing results from all clustering methods
  """
  if layer_name not in self.activations:
   raise ValueError(f"Layer {layer_name} not found in activations")
   
  activation = self.activations[layer_name]
  
  # Reshape and prepare data
  if activation.ndim > 2:
   n_samples = activation.shape[0]
   n_features = np.prod(activation.shape[1:])
   data = activation.reshape(n_samples, n_features)
  else:
   data = activation
  
  # Standardize data
  scaler = StandardScaler()
  data_scaled = scaler.fit_transform(data)
  
  clustering_results = {}
  
  for method in methods:
   try:
    if method == 'kmeans':
     # K-means with automatic k selection
     silhouette_scores = []
     k_range = range(2, min(10, data.shape[0]))
     
     for k in k_range:
      kmeans = KMeans(n_clusters=k, random_state=42)
      labels = kmeans.fit_predict(data_scaled)
      if len(np.unique(labels)) > 1:
       score = silhouette_score(data_scaled, labels)
       silhouette_scores.append(score)
      else:
       silhouette_scores.append(-1)
     
     if silhouette_scores:
      best_k = k_range[np.argmax(silhouette_scores)]
      kmeans = KMeans(n_clusters=best_k, random_state=42)
      labels = kmeans.fit_predict(data_scaled)
      
      clustering_results['kmeans'] = {
       'labels': labels,
       'n_clusters': best_k,
       'silhouette_score': max(silhouette_scores),
       'cluster_centers': kmeans.cluster_centers_,
       'inertia': kmeans.inertia_
      }
     
    elif method == 'dbscan':
     # DBSCAN with automatic eps selection
     eps_range = np.linspace(0.1, 2.0, 10)
     best_eps = eps_range[0]
     best_score = -1
     
     for eps in eps_range:
      dbscan = DBSCAN(eps=eps, min_samples=max(2, data.shape[0] // 10))
      labels = dbscan.fit_predict(data_scaled)
      
      if len(np.unique(labels)) > 1 and -1 not in labels:
       score = silhouette_score(data_scaled, labels)
       if score > best_score:
        best_score = score
        best_eps = eps
     
     dbscan = DBSCAN(eps=best_eps, min_samples=max(2, data.shape[0] // 10))
     labels = dbscan.fit_predict(data_scaled)
     
     clustering_results['dbscan'] = {
      'labels': labels,
      'eps': best_eps,
      'n_clusters': len(np.unique(labels)) - (1 if -1 in labels else 0),
      'n_noise': int(np.sum(labels == -1)),
      'silhouette_score': best_score if best_score > -1 else None
     }
     
    elif method == 'hierarchical':
     from sklearn.cluster import AgglomerativeClustering
     
     # Hierarchical clustering with automatic n_clusters
     silhouette_scores = []
     k_range = range(2, min(10, data.shape[0]))
     
     for k in k_range:
      hierarchical = AgglomerativeClustering(n_clusters=k)
      labels = hierarchical.fit_predict(data_scaled)
      if len(np.unique(labels)) > 1:
       score = silhouette_score(data_scaled, labels)
       silhouette_scores.append(score)
      else:
       silhouette_scores.append(-1)
     
     if silhouette_scores:
      best_k = k_range[np.argmax(silhouette_scores)]
      hierarchical = AgglomerativeClustering(n_clusters=best_k)
      labels = hierarchical.fit_predict(data_scaled)
      
      clustering_results['hierarchical'] = {
       'labels': labels,
       'n_clusters': best_k,
       'silhouette_score': max(silhouette_scores)
      }
     
   except Exception as e:
    logger.warning(f"Error in {method} clustering: {str(e)}")
    continue
  
  return clustering_results

 def export_detailed_report(self, output_dir: str) -> Dict[str, str]:
  """
  Export a comprehensive analysis report with visualizations.
  
  Args:
   output_dir: Directory to save all report files
   
  Returns:
   Dictionary mapping report component names to file paths
  """
  if not self.activations:
   raise ValueError("No activations available. Run simulate() first.")
   
  # Create output directory
  os.makedirs(output_dir, exist_ok=True)
  
  # Generate all analyses
  if not self.layer_metrics:
   self.analyze_distribution()
   
  stats = self.compute_activation_statistics()
  report = self.generate_activation_report()
  
  # File paths
  files_created = {}
  
  # 1. Main report (JSON)
  report_path = os.path.join(output_dir, 'activation_report.json')
  with open(report_path, 'w') as f:
   json.dump(report, f, indent=2, default=str)
  files_created['main_report'] = report_path
  
  # 2. Detailed statistics (JSON)
  stats_path = os.path.join(output_dir, 'detailed_statistics.json')
  with open(stats_path, 'w') as f:
   json.dump(stats, f, indent=2, default=str)
  files_created['detailed_stats'] = stats_path
  
  # 3. Activation heatmap
  heatmap_path = os.path.join(output_dir, 'activation_heatmap.png')
  self.visualize_activation_heatmap(save_path=heatmap_path)
  files_created['heatmap'] = heatmap_path
  
  # 4. Distribution plots for each layer
  dist_dir = os.path.join(output_dir, 'distributions')
  os.makedirs(dist_dir, exist_ok=True)
  
  for layer_name in self.activations.keys():
   dist_path = os.path.join(dist_dir, f'{layer_name}_distribution.png')
   self.visualize_distribution(layer_name, save_path=dist_path)
   files_created[f'distribution_{layer_name}'] = dist_path
  
  # 5. Layer comparison plots
  comparison_dir = os.path.join(output_dir, 'comparisons')
  os.makedirs(comparison_dir, exist_ok=True)
  
  metrics_to_compare = ['mean', 'std_dev', 'sparsity', 'skewness', 'kurtosis']
  layer_names = list(self.activations.keys())
  
  for metric in metrics_to_compare:
   comp_path = os.path.join(comparison_dir, f'{metric}_comparison.png')
   try:
    self.visualize_layer_comparison(layer_names, metric=metric, save_path=comp_path)
    files_created[f'comparison_{metric}'] = comp_path
   except Exception as e:
    logger.warning(f"Could not create comparison plot for {metric}: {str(e)}")
  
  # 6. Export raw activations
  activations_path = os.path.join(output_dir, 'activations.h5')
  self.export_activations(activations_path, format='hdf5', include_metadata=True)
  files_created['activations'] = activations_path
  
  # 7. Create summary README
  readme_path = os.path.join(output_dir, 'README.md')
  with open(readme_path, 'w') as f:
   f.write("# Activation Analysis Report\n\n")
   f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
   f.write(f"## Summary\n")
   f.write(f"- Total layers analyzed: {len(self.activations)}\n")
   f.write(f"- Total parameters: {sum(act.size for act in self.activations.values()):,}\n\n")
   f.write("## Files Generated\n")
   for name, path in files_created.items():
    f.write(f"- **{name}**: {os.path.basename(path)}\n")
   f.write("\n## Layer Overview\n")
   for layer_name in self.activations.keys():
    summary = self.get_layer_summary(layer_name)
    f.write(f"### {layer_name}\n")
    f.write(f"- Shape: {summary['shape']}\n")
    f.write(f"- Parameters: {summary['total_parameters']:,}\n")
    if 'mean_activation' in summary:
     f.write(f"- Mean activation: {summary['mean_activation']:.4f}\n")
     f.write(f"- Sparsity: {summary['sparsity']:.4f}\n")
    f.write("\n")
  
  files_created['readme'] = readme_path
  
  logger.info(f"Exported detailed report to {output_dir} with {len(files_created)} files")
  return files_created
   