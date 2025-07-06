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
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
import os
from pathlib import Path

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
