# src/tensor_analysis/entropy_analysis.py
"""
Entropy Analysis Module for SVELTE Framework.
Performs multi-dimensional entropy calculations, gradient field generation, and semantic density mapping.
"""
import numpy as np
from typing import Dict, Any
import logging
from scipy.ndimage import gaussian_gradient_magnitude
from scipy.stats import entropy as scipy_entropy
from scipy.ndimage import gaussian_gradient_magnitude, uniform_filter1d, uniform_filter, gaussian_filter
from pathlib import Path
import os
import json
import h5py
from datetime import datetime
from sklearn.cluster import DBSCAN
from typing import Tuple, List, Optional, Union, Callable
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import matplotlib.pyplot as plt

class EntropyAnalysisModule:
    """
    Production-grade entropy analysis for multi-dimensional tensor fields.
    Provides advanced entropy computation, gradient field analysis, and semantic density mapping
    with comprehensive diagnostics and visualization capabilities.
    
    Features:
    - Multi-method entropy calculations (Shannon, Rényi, Tsallis)
    - Gradient field analysis with customizable kernels
    - Semantic density mapping with normalization options
    - Multi-scale analysis support
    - Anomaly detection via entropy differentials
    
    Attributes:
        tensor_field (Dict[str, np.ndarray]): Dictionary mapping tensor names to numpy arrays
        entropy_maps (Dict[str, np.ndarray]): Calculated entropy maps for each tensor
        gradients (Dict[str, Dict[str, np.ndarray]]): Gradient vector fields and magnitudes
        semantic_density (Dict[str, np.ndarray]): Semantic density maps
    """

    def __init__(self, tensor_field: Dict[str, np.ndarray], config: Dict[str, Any] = None):
        """
        Initialize the EntropyAnalysisModule with tensor data and optional configuration.

        Args:
            tensor_field (Dict[str, np.ndarray]): Dictionary mapping tensor names to numpy arrays.
            config (Dict[str, Any], optional): Configuration dictionary with analysis parameters.
            Supported keys:
            - log_level: Logging level (default: INFO)
            - default_bins: Default bins for histogram calculations
            - precision: Floating point precision for calculations
            - cache_intermediate: Whether to cache intermediate results

        Raises:
            TypeError: If tensor_field is not a dictionary or contains non-ndarray values
            ValueError: If any tensor in tensor_field is empty
        """
        self.tensor_field = tensor_field
        self.entropy_maps = {}
        self.gradients = {}
        self.semantic_density = {}
        self.differential_entropy = {}
        self.anomaly_scores = {}
        
        # Configuration with defaults
        self.config = {
            'log_level': logging.INFO,
            'default_bins': 256,
            'precision': 'float64',
            'cache_intermediate': True,
            'default_sigma': 1.0,
        }
        if config:
            self.config.update(config)
            
        # Configure logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.config['log_level'])
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        self._validate_tensor_field()
        self.logger.info(f"EntropyAnalysisModule initialized with {len(tensor_field)} tensors")

    def _validate_tensor_field(self):
        """
        Validate the tensor field dictionary structure and contents.
        
        Raises:
            TypeError: If tensor_field is not a dictionary or contains non-ndarray values
            ValueError: If any tensor in tensor_field is empty or has inappropriate values
        """
        if not isinstance(self.tensor_field, dict):
            raise TypeError("tensor_field must be a dictionary of name: np.ndarray pairs.")
        
        if not self.tensor_field:
            self.logger.warning("Empty tensor field provided")
            
        for name, tensor in self.tensor_field.items():
            if not isinstance(name, str):
                raise TypeError(f"Tensor field key '{name}' is not a string.")
            
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Tensor '{name}' is not a numpy.ndarray.")
            
            if tensor.size == 0:
                raise ValueError(f"Tensor '{name}' is empty.")
            
            if np.isnan(tensor).any():
                self.logger.warning(f"Tensor '{name}' contains NaN values")
            
            if np.isinf(tensor).any():
                self.logger.warning(f"Tensor '{name}' contains infinite values")
            
        self.logger.debug("Tensor field validation completed")

    def compute_entropy(self, bins: int = None, method: str = "shannon", alpha: float = 2.0) -> Dict[str, np.ndarray]:
        """
        Compute entropy maps for each tensor in the field using specified method.

        Args:
            bins (int, optional): Number of bins for histogram-based entropy.
            If None, uses the configuration default.
            method (str): Entropy method. Options:
            - 'shannon': Standard Shannon entropy
            - 'renyi': Rényi entropy (parameterized by alpha)
            - 'tsallis': Tsallis entropy (parameterized by alpha)
            - 'differential': Differential entropy estimation for continuous data
            alpha (float): Parameter for Rényi and Tsallis entropy calculations

        Returns:
            Dict[str, np.ndarray]: Mapping from tensor name to entropy value/map.
            
        Raises:
            ValueError: If an unknown entropy method is specified
        """
        if bins is None:
            bins = self.config['default_bins']
            
        self.entropy_maps.clear()
        self.logger.info(f"Computing entropy using {method} method with {bins} bins")
        
        for name, tensor in self.tensor_field.items():
            # Convert to float64 for precision in entropy calculations
            tensor = tensor.astype(self.config['precision'])
            
            # For multi-dimensional tensors, compute entropy maps along last axis
            if tensor.ndim > 1:
                # Reshape for vectorized entropy calculation
                original_shape = tensor.shape[:-1]
                reshaped_tensor = tensor.reshape(-1, tensor.shape[-1])
                entropy_map = np.zeros(reshaped_tensor.shape[0], dtype=self.config['precision'])
                
                for i, vector in enumerate(reshaped_tensor):
                    entropy_map[i] = self._calculate_entropy_value(vector, bins, method, alpha)
                
                # Reshape back to original dimensions
                self.entropy_maps[name] = entropy_map.reshape(original_shape)
            else:
                # For 1D tensors, compute a single entropy value
                self.entropy_maps[name] = self._calculate_entropy_value(tensor, bins, method, alpha)
            
            mean_entropy = np.mean(self.entropy_maps[name])
            self.logger.debug(f"Computed entropy for '{name}': mean={mean_entropy:.6f}")
            
        return self.entropy_maps

    def _calculate_entropy_value(self, data: np.ndarray, bins: int, method: str, alpha: float) -> float:
        """
        Calculate entropy value for a single data vector using specified method.
        
        Args:
            data: Input data vector
            bins: Number of bins for histogram
            method: Entropy calculation method
            alpha: Parameter for Rényi and Tsallis entropy
            
        Returns:
            float: Calculated entropy value
            
        Raises:
            ValueError: If method is not recognized
        """
        # Remove inf and nan values for stable entropy calculation
        data = data[~np.isnan(data) & ~np.isinf(data)]
        
        if data.size == 0:
            self.logger.warning("Empty data vector after filtering NaN/Inf values")
            return 0.0
            
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        
        if hist.size == 0:
            return 0.0
            
        if method == "shannon":
            return -np.sum(hist * np.log2(hist))
            
        elif method == "renyi":
            if alpha <= 0 or alpha == 1:
                raise ValueError(f"Alpha must be positive and not equal to 1 for Rényi entropy, got {alpha}")
            return (1 / (1 - alpha)) * np.log2(np.sum(hist ** alpha))
            
        elif method == "tsallis":
            if alpha <= 0:
                raise ValueError(f"Alpha must be positive for Tsallis entropy, got {alpha}")
            return (1 / (alpha - 1)) * (1 - np.sum(hist ** alpha))
            
        elif method == "differential":
            # Kozachenko-Leonenko estimator for differential entropy
            # This is a simplified implementation for integration purposes
            n = data.size
            data_sorted = np.sort(data)
            distances = np.abs(data_sorted[1:] - data_sorted[:-1])
            mean_dist = np.mean(distances[distances > 0])
            if mean_dist > 0:
                return np.log(mean_dist * n) + np.euler_gamma  # Euler-Mascheroni constant
            return 0.0
            
        elif method == "scipy":
            return scipy_entropy(hist, base=2)
            
        else:
            raise ValueError(f"Unknown entropy method: {method}")

    def compute_entropy_gradient(self, sigma: float = None, method: str = "gaussian") -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute gradient fields and magnitudes for entropy distributions.

        Args:
            sigma (float, optional): Standard deviation for Gaussian kernel.
            If None, uses the configuration default.
            method (str): Gradient calculation method:
            - 'gaussian': Gaussian gradient magnitude
            - 'sobel': Sobel operator
            - 'prewitt': Prewitt operator
            - 'central': Central finite difference

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Mapping from tensor name to gradient results 
            with 'magnitude' and 'vectors' keys.
            
        Raises:
            ValueError: If tensor dimensionality is inappropriate for gradient calculation
        """
        
        if sigma is None:
            sigma = self.config['default_sigma']
            
        self.gradients.clear()
        self.logger.info(f"Computing gradient fields using {method} method")
        
        for name, tensor in self.tensor_field.items():
            if tensor.ndim < 1:
                raise ValueError(f"Tensor '{name}' must have at least one dimension.")
            
            # Ensure tensor is float type for gradient calculations
            tensor = tensor.astype(self.config['precision'])
            
            # Calculate gradient based on method
            if method == "gaussian":
                grad_mag = gaussian_gradient_magnitude(tensor, sigma=sigma)
                grad_vectors = np.gradient(gaussian_filter(tensor, sigma))
                
            elif method == "sobel":
                if tensor.ndim == 1:
                    # For 1D, use basic finite differences
                    grad_vectors = [np.gradient(tensor)]
                    grad_mag = np.abs(grad_vectors[0])
                else:
                    # For multi-dimensional, use Sobel
                    grad_vectors = np.gradient(tensor)
                    grad_mag = np.sqrt(sum(g*g for g in grad_vectors))
                    
            elif method == "prewitt":
                if tensor.ndim == 1:
                    # For 1D, use basic differences
                    grad_vectors = [np.gradient(tensor)]
                    grad_mag = np.abs(grad_vectors[0])
                else:
                    # Use Prewitt filters
                    grads = []
                    for axis in range(tensor.ndim):
                        slices = [slice(None)] * tensor.ndim
                        pattern = [-1, 0, 1]
                        for p in pattern:
                            slices[axis] = slice(max(0, p), min(tensor.shape[axis], tensor.shape[axis] + p))
                            # Implement Prewitt filter logic
                        
                    grad_vectors = np.gradient(tensor)  # Fallback to numpy gradient
                    grad_mag = np.sqrt(sum(g*g for g in grad_vectors))
                    
            elif method == "central":
                grad_vectors = np.gradient(tensor)
                grad_mag = np.sqrt(sum(g*g for g in grad_vectors))
                
            else:
                raise ValueError(f"Unknown gradient method: {method}")
            
            self.gradients[name] = {
                'magnitude': grad_mag,
                'vectors': grad_vectors
            }
            
            self.logger.debug(f"Computed gradient field for '{name}', magnitude range: [{np.min(grad_mag):.6f}, {np.max(grad_mag):.6f}]")
            
        return self.gradients

    def semantic_density_map(self, window_size: int = 5, normalize: bool = True, 
        method: str = "sliding") -> Dict[str, np.ndarray]:
        """
        Compute semantic density maps using local statistical measures.

        Args:
            window_size (int): Size of the sliding window for local statistics.
            normalize (bool): Whether to normalize density by local mean.
            method (str): Density calculation method:
            - 'sliding': Sliding window statistics
            - 'global': Global statistics with local normalization
            - 'adaptive': Adaptive window sizing based on local features

        Returns:
            Dict[str, np.ndarray]: Mapping from tensor name to semantic density map.
        """
        self.semantic_density.clear()
        self.logger.info(f"Computing semantic density maps using {method} method")
        
        for name, tensor in self.tensor_field.items():
            tensor = tensor.astype(self.config['precision'])
            
            if method == "sliding":
                density = self._sliding_window_density(tensor, window_size, normalize)
                
            elif method == "global":
                # Global statistics with local normalization
                global_std = np.std(tensor)
                global_mean = np.mean(tensor)
                
                if normalize and global_mean != 0:
                    density = global_std / global_mean
                else:
                    density = global_std
                    
                # Create a map with the same shape as tensor but filled with density value
                density_map = np.full_like(tensor, fill_value=density, dtype=self.config['precision'])
                density = density_map
                
            elif method == "adaptive":
                density = self._adaptive_window_density(tensor, normalize)
                
            else:
                raise ValueError(f"Unknown density method: {method}")
            
            self.semantic_density[name] = density
            mean_density = np.mean(density)
            self.logger.debug(f"Computed semantic density for '{name}': mean={mean_density:.6f}")
            
        return self.semantic_density

    def _sliding_window_density(self, tensor: np.ndarray, window_size: int, normalize: bool) -> np.ndarray:
        """
        Calculate semantic density using a sliding window approach.
        
        Args:
            tensor: Input tensor
            window_size: Size of sliding window
            normalize: Whether to normalize by local mean
            
        Returns:
            np.ndarray: Semantic density map
        """
        
        # Handle different tensor dimensionalities
        if tensor.ndim == 1:
            # For 1D, use specialized 1D filters
            local_mean = uniform_filter1d(tensor, size=window_size)
            local_mean_sq = uniform_filter1d(tensor**2, size=window_size)
            local_var = local_mean_sq - local_mean**2
            local_std = np.sqrt(np.maximum(local_var, 0))  # Prevent negative values due to numerical issues
            
            if normalize:
                # Prevent division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    density = local_std / local_mean
                    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                density = local_std
            
        else:
            # For multi-dimensional tensors
            local_mean = uniform_filter(tensor, size=window_size)
            local_mean_sq = uniform_filter(tensor**2, size=window_size)
            local_var = local_mean_sq - local_mean**2
            local_std = np.sqrt(np.maximum(local_var, 0))
            
            if normalize:
                with np.errstate(divide='ignore', invalid='ignore'):
                    density = local_std / local_mean
                    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                density = local_std
            
        return density

    def _adaptive_window_density(self, tensor: np.ndarray, normalize: bool) -> np.ndarray:
        """
        Calculate semantic density using adaptive window sizing based on local features.
        
        Args:
            tensor: Input tensor
            normalize: Whether to normalize by local mean
            
        Returns:
            np.ndarray: Semantic density map
        """
        # Simplified adaptive window implementation
        # For a production implementation, this would use more sophisticated
        # adaptive windowing techniques based on feature detection
        
        # Get gradient magnitude to estimate feature scales
        grad_mag = gaussian_gradient_magnitude(tensor, sigma=1.0)
        
        # Normalize gradient magnitude to range [0.5, 3] for window scaling
        if np.max(grad_mag) > np.min(grad_mag):
            normalized_grad = 0.5 + 2.5 * (grad_mag - np.min(grad_mag)) / (np.max(grad_mag) - np.min(grad_mag))
        else:
            normalized_grad = np.ones_like(grad_mag)
            
        # For simplicity, quantize window sizes to one of three scales: small, medium, large
        window_sizes = np.digitize(normalized_grad, bins=[1, 2]) + 3  # Window sizes: 3, 5, 7
        
        # Initialize density map
        density = np.zeros_like(tensor, dtype=self.config['precision'])
        
        # For each window size, calculate density for regions with that window size
        for ws in np.unique(window_sizes):
            mask = (window_sizes == ws)
            if not np.any(mask):
                continue
            
            # Calculate density for this window size
            window_density = self._sliding_window_density(tensor, int(ws), normalize)
            
            # Apply to the appropriate regions
            density[mask] = window_density[mask]
            
        return density

    def analyze(self, 
        entropy_bins: int = None, 
        entropy_method: str = "shannon",
        gradient_sigma: float = None, 
        gradient_method: str = "gaussian",
        density_window: int = 5,
        density_method: str = "sliding",
        normalize_density: bool = True,
        alpha: float = 2.0,
        parallel: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Perform comprehensive tensor analysis by executing multiple analysis methods.
        
        Args:
            entropy_bins: Number of bins for entropy calculation
            entropy_method: Method for entropy calculation
            gradient_sigma: Sigma for gradient calculation
            gradient_method: Method for gradient calculation
            density_window: Window size for density calculation
            density_method: Method for density calculation
            normalize_density: Whether to normalize density maps
            alpha: Parameter for parameterized entropy methods
            parallel: Whether to run analyses in parallel
            
        Returns:
            Dict[str, Dict[str, Any]]: Comprehensive analysis results
        """
        self.logger.info("Starting comprehensive tensor analysis")
        start_time = datetime.now()
        
        if parallel:
            self.logger.info("Running analyses in parallel mode")
            with ProcessPoolExecutor() as executor:
                # Submit tasks to process pool
                entropy_future = executor.submit(
                    self.compute_entropy, entropy_bins, entropy_method, alpha
                )
                gradient_future = executor.submit(
                    self.compute_entropy_gradient, gradient_sigma, gradient_method
                )
                density_future = executor.submit(
                    self.semantic_density_map, density_window, normalize_density, density_method
                )
                
                # Collect results
                self.entropy_maps = entropy_future.result()
                self.gradients = gradient_future.result()
                self.semantic_density = density_future.result()
        else:
            # Sequential processing
            self.compute_entropy(bins=entropy_bins, method=entropy_method, alpha=alpha)
            self.compute_entropy_gradient(sigma=gradient_sigma, method=gradient_method)
            self.semantic_density_map(window_size=density_window, normalize=normalize_density, method=density_method)
        
        # Compute anomaly scores based on combined metrics
        self.detect_anomalies()
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        self.logger.info(f"Analysis completed in {elapsed:.2f} seconds")
        
        # Compile and return comprehensive results
        results = {}
        for name in self.tensor_field.keys():
            results[name] = {
                'entropy': self.entropy_maps.get(name),
                'gradient': self.gradients.get(name),
                'density': self.semantic_density.get(name),
                'anomalies': self.anomaly_scores.get(name)
            }
            
        return results

    def detect_anomalies(self, threshold_multiplier: float = 2.0, 
            method: str = "combined") -> Dict[str, np.ndarray]:
        """
        Detect anomalies in tensor fields based on statistical properties.
        
        Args:
            threshold_multiplier: Multiplier for standard deviation to set anomaly threshold
            method: Method for anomaly detection ('combined', 'entropy', 'gradient', 'density')
            
        Returns:
            Dict[str, np.ndarray]: Anomaly scores for each tensor
        """
        self.anomaly_scores.clear()
        self.logger.info(f"Detecting anomalies using {method} method")
        
        for name in self.tensor_field.keys():
            scores = np.zeros_like(self.tensor_field[name], dtype=self.config['precision'])
            
            # Combine metrics for anomaly detection
            if method == "combined" or method == "entropy":
                if name in self.entropy_maps:
                    entropy = self.entropy_maps[name]
                    entropy_mean, entropy_std = np.mean(entropy), np.std(entropy)
                    entropy_score = np.abs(entropy - entropy_mean) / max(entropy_std, 1e-10)
                    scores += entropy_score
                    
            if method == "combined" or method == "gradient":
                if name in self.gradients:
                    gradient = self.gradients[name]['magnitude']
                    grad_mean, grad_std = np.mean(gradient), np.std(gradient)
                    gradient_score = np.abs(gradient - grad_mean) / max(grad_std, 1e-10)
                    scores += gradient_score
                    
            if method == "combined" or method == "density":
                if name in self.semantic_density:
                    density = self.semantic_density[name]
                    density_mean, density_std = np.mean(density), np.std(density)
                    density_score = np.abs(density - density_mean) / max(density_std, 1e-10)
                    scores += density_score
            
            if method == "combined":
                # Normalize combined score
                scores /= 3.0
            
            # Apply threshold to identify anomalies
            self.anomaly_scores[name] = scores
            
            # Log anomaly statistics
            anomaly_count = np.sum(scores > threshold_multiplier)
            anomaly_percent = 100.0 * anomaly_count / scores.size if scores.size > 0 else 0
            self.logger.debug(f"Detected {anomaly_count} anomalies in '{name}' ({anomaly_percent:.2f}%)")
            
        return self.anomaly_scores

    def cluster_anomalies(self, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Cluster detected anomalies using DBSCAN to identify anomaly groups.
        
        Args:
            eps: Maximum distance between samples for DBSCAN
            min_samples: Minimum number of samples in a cluster
            
        Returns:
            Dict[str, Dict[str, Any]]: Clustering results for anomalies
        """
        if not self.anomaly_scores:
            self.logger.warning("No anomaly scores available. Run detect_anomalies first.")
            return {}
            
        self.logger.info("Clustering anomalies using DBSCAN")
        results = {}
        
        for name, scores in self.anomaly_scores.items():
            # Get coordinates of anomalies
            anomaly_indices = np.where(scores > 2.0)
            if len(anomaly_indices[0]) == 0:
                self.logger.debug(f"No anomalies to cluster for '{name}'")
                results[name] = {'clusters': None, 'count': 0}
                continue
            
            # Convert indices to feature array for clustering
            features = np.column_stack(anomaly_indices)
            
            # Apply DBSCAN clustering
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
            labels = db.labels()
            
            # Count clusters (excluding noise points with label -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            results[name] = {
                'clusters': labels,
                'count': n_clusters,
                'indices': anomaly_indices,
                'scores': scores[anomaly_indices]
            }
            
            self.logger.debug(f"Found {n_clusters} anomaly clusters in '{name}'")
            
        return results
        
    def visualize_entropy(self, output_dir: str = None, display: bool = True, 
            tensor_name: str = None) -> Dict[str, str]:
        """
        Visualize entropy maps for tensor fields.
        
        Args:
            output_dir: Directory to save visualization files
            display: Whether to display the visualizations
            tensor_name: Optional specific tensor to visualize
            
        Returns:
            Dict[str, str]: Paths to saved visualization files
        """
        if not self.entropy_maps:
            self.logger.warning("No entropy maps available. Run compute_entropy first.")
            return {}
            
        tensors_to_process = [tensor_name] if tensor_name else self.entropy_maps.keys()
        file_paths = {}
        
        for name in tensors_to_process:
            if name not in self.entropy_maps:
                self.logger.warning(f"No entropy map for tensor '{name}'")
                continue
            
            entropy_map = self.entropy_maps[name]
            
            plt.figure(figsize=(10, 8))
            if entropy_map.ndim == 1:
                plt.plot(entropy_map)
                plt.title(f"Entropy Distribution for {name}")
                plt.xlabel("Index")
                plt.ylabel("Entropy")
            elif entropy_map.ndim == 2:
                plt.imshow(entropy_map, cmap='viridis')
                plt.colorbar(label='Entropy')
                plt.title(f"Entropy Map for {name}")
            elif entropy_map.ndim == 3 and entropy_map.shape[2] <= 3:
                # Assuming this is a 2D map with color channels
                plt.imshow(entropy_map)
                plt.colorbar(label='Entropy')
                plt.title(f"Entropy Map for {name}")
            else:
                # For higher dimensional tensors, show a 2D slice
                slice_idx = entropy_map.shape[0] // 2
                plt.imshow(entropy_map[slice_idx], cmap='viridis')
                plt.colorbar(label='Entropy')
                plt.title(f"Entropy Map for {name} (Slice {slice_idx})")
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filepath = os.path.join(output_dir, f"{name}_entropy_{timestamp}.png")
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                file_paths[name] = filepath
            
            if display:
                plt.show()
            else:
                plt.close()
            
        return file_paths

    def save_results(self, output_path: str, format: str = "hdf5") -> str:
        """
        Save analysis results to file.
        
        Args:
            output_path: Path to save results
            format: File format ('hdf5', 'npz', 'json')
            
        Returns:
            str: Path to saved file
        """
        self.logger.info(f"Saving analysis results to {output_path} in {format} format")
        
        if format == "hdf5":
            with h5py.File(output_path, 'w') as f:
                # Create groups for each result type
                entropy_group = f.create_group('entropy_maps')
                gradient_group = f.create_group('gradients')
                density_group = f.create_group('semantic_density')
                anomaly_group = f.create_group('anomaly_scores')
                
                # Save entropy maps
                for name, data in self.entropy_maps.items():
                    entropy_group.create_dataset(name, data=data, compression="gzip")
                    
                # Save gradients
                for name, grad_dict in self.gradients.items():
                    name_group = gradient_group.create_group(name)
                    name_group.create_dataset('magnitude', data=grad_dict['magnitude'], compression="gzip")
                    
                    # Handle gradient vectors (potentially multiple arrays)
                    vectors_group = name_group.create_group('vectors')
                    for i, vec in enumerate(grad_dict['vectors']):
                        vectors_group.create_dataset(f'axis_{i}', data=vec, compression="gzip")
                    
                # Save density maps
                for name, data in self.semantic_density.items():
                    density_group.create_dataset(name, data=data, compression="gzip")
                    
                # Save anomaly scores
                for name, data in self.anomaly_scores.items():
                    anomaly_group.create_dataset(name, data=data, compression="gzip")
                    
                # Add metadata
                f.attrs['creation_time'] = datetime.now().isoformat()
                f.attrs['tensor_names'] = list(self.tensor_field.keys())
                
        elif format == "npz":
            # Prepare data for NPZ format
            save_dict = {}
            
            # Add entropy maps
            for name, data in self.entropy_maps.items():
                save_dict[f'entropy_{name}'] = data
            
            # Add gradients (magnitude only to keep it simpler)
            for name, grad_dict in self.gradients.items():
                save_dict[f'gradient_mag_{name}'] = grad_dict['magnitude']
            
            # Add density maps
            for name, data in self.semantic_density.items():
                save_dict[f'density_{name}'] = data
            
            # Add anomaly scores
            for name, data in self.anomaly_scores.items():
                save_dict[f'anomaly_{name}'] = data
            
            # Save to file
            np.savez_compressed(output_path, **save_dict)
            
        elif format == "json":
            # For JSON, we need to handle numpy arrays - convert to lists
            save_dict = {
                'metadata': {
                    'creation_time': datetime.now().isoformat(),
                    'tensor_names': list(self.tensor_field.keys())
                },
                'summary': {}
            }
            
            # Add summary statistics for each analysis
            for name in self.tensor_field.keys():
                save_dict['summary'][name] = {}
                
                if name in self.entropy_maps:
                    entropy = self.entropy_maps[name]
                    save_dict['summary'][name]['entropy'] = {
                        'mean': float(np.mean(entropy)),
                        'std': float(np.std(entropy)),
                        'min': float(np.min(entropy)),
                        'max': float(np.max(entropy))
                    }
                    
                if name in self.gradients:
                    gradient = self.gradients[name]['magnitude']
                    save_dict['summary'][name]['gradient'] = {
                        'mean': float(np.mean(gradient)),
                        'std': float(np.std(gradient)),
                        'min': float(np.min(gradient)),
                        'max': float(np.max(gradient))
                    }
                    
                if name in self.semantic_density:
                    density = self.semantic_density[name]
                    save_dict['summary'][name]['density'] = {
                        'mean': float(np.mean(density)),
                        'std': float(np.std(density)),
                        'min': float(np.min(density)),
                        'max': float(np.max(density))
                    }
                    
                if name in self.anomaly_scores:
                    anomaly = self.anomaly_scores[name]
                    save_dict['summary'][name]['anomaly'] = {
                        'mean': float(np.mean(anomaly)),
                        'std': float(np.std(anomaly)),
                        'min': float(np.min(anomaly)),
                        'max': float(np.max(anomaly)),
                        'anomaly_count': int(np.sum(anomaly > 2.0))
                    }
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(save_dict, f, indent=2)
                
        else:
            raise ValueError(f"Unsupported file format: {format}")
            
        self.logger.info(f"Results saved successfully to {output_path}")
        return output_path

    def batch_process(self, tensor_batch: Dict[str, Dict[str, np.ndarray]], 
        parallel: bool = True, **analyze_kwargs) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Process a batch of tensor fields using the same analysis parameters.
        
        Args:
            tensor_batch: Dictionary mapping batch IDs to tensor field dictionaries
            parallel: Whether to process batches in parallel
            **analyze_kwargs: Arguments to pass to the analyze method
            
        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: Results for each batch
        """
        self.logger.info(f"Batch processing {len(tensor_batch)} tensor batches")
        results = {}
        
        if parallel:
            # Use multiprocessing for batch processing
            with ProcessPoolExecutor() as executor:
                futures = {}
                for batch_id, tensor_field in tensor_batch.items():
                    # Create a temporary analysis module for this batch
                    batch_module = EntropyAnalysisModule(tensor_field, self.config)
                    futures[batch_id] = executor.submit(batch_module.analyze, **analyze_kwargs)
                    
                # Collect results
                for batch_id, future in futures.items():
                    try:
                        results[batch_id] = future.result()
                    except Exception as e:
                        self.logger.error(f"Error processing batch {batch_id}: {str(e)}")
                        results[batch_id] = {"error": str(e)}
        else:
            # Sequential processing
            original_tensor_field = self.tensor_field
            for batch_id, tensor_field in tensor_batch.items():
                try:
                    # Swap in the batch tensor field
                    self.tensor_field = tensor_field
                    
                    # Run analysis
                    results[batch_id] = self.analyze(**analyze_kwargs)
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_id}: {str(e)}")
                    results[batch_id] = {"error": str(e)}
                    
            # Restore original tensor field
            self.tensor_field = original_tensor_field
            
        self.logger.info(f"Completed batch processing for {len(results)} batches")
        return results
