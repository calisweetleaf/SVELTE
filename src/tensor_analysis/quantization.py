# src/tensor_analysis/quantization.py
"""
Quantization scheme identification and dequantization simulation for SVELTE Framework.
"""
import numpy as np
from typing import Dict, Any
import warnings
from scipy import stats as scipy_stats # Added for skew/kurtosis
from scipy import signal # Added for find_peaks

class QuantizationReconstructor:
    def __init__(self, quantization_info: Dict[str, Any]):
        self.quantization_info = quantization_info

    def simulate_dequantization(self, tensor: np.ndarray, layer_name: str = None) -> np.ndarray:
     """
     Simulates the dequantization process on a quantized tensor based on quantization_info.
     
     Args:
      tensor (np.ndarray): The quantized tensor to dequantize
      layer_name (str, optional): The name of the layer for layer-specific quantization params
     
     Returns:
      np.ndarray: The dequantized tensor in floating point format
      
     Raises:
      ValueError: If quantization scheme is not supported or parameters are missing
     """
     # Extract quantization scheme from info
     scheme = self.quantization_info.get("scheme", "unknown")
     bits = self.quantization_info.get("bits", 8)
     
     # Layer-specific parameters take precedence if available
     if layer_name and layer_name in self.quantization_info.get("layer_specific", {}):
      layer_params = self.quantization_info["layer_specific"][layer_name]
      scheme = layer_params.get("scheme", scheme)
      bits = layer_params.get("bits", bits)
     
     # Convert tensor to proper dtype for processing if needed
     if tensor.dtype != np.int8 and tensor.dtype != np.int16 and tensor.dtype != np.int32:
      if bits <= 8:
       tensor = tensor.astype(np.int8)
      elif bits <= 16:
       tensor = tensor.astype(np.int16)
      else:
       tensor = tensor.astype(np.int32)
     
     # Handle different quantization schemes
     if scheme == "symmetric_uniform":
      return self._symmetric_uniform_dequantize(tensor, bits, layer_name)
     elif scheme == "asymmetric_uniform":
      return self._asymmetric_uniform_dequantize(tensor, bits, layer_name)
     elif scheme == "blockwise":
      return self._blockwise_dequantize(tensor, bits, layer_name)
     elif scheme == "grouped":
      return self._grouped_dequantize(tensor, bits, layer_name)
     elif scheme == "k_quants":
      return self._k_quants_dequantize(tensor, bits, layer_name)
     elif scheme == "custom":
      return self._custom_dequantize(tensor, layer_name)
     else:
      # Fallback to identity mapping with warning
      warnings.warn(f"Unsupported quantization scheme: {scheme}. Returning original tensor.")
      if tensor.dtype not in [np.float16, np.float32, np.float64]:
       return tensor.astype(np.float32)
      return tensor

    def _symmetric_uniform_dequantize(self, tensor: np.ndarray, bits: int, layer_name: str = None) -> np.ndarray:
     """Symmetric uniform dequantization (zero-point = 0)"""
     scale = self._get_scale_parameter(bits, layer_name)
     
     # Apply dequantization: q = r / scale
     dequantized = tensor.astype(np.float32) * scale
     return dequantized

    def _asymmetric_uniform_dequantize(self, tensor: np.ndarray, bits: int, layer_name: str = None) -> np.ndarray:
     """Asymmetric uniform dequantization with zero-point correction"""
     scale = self._get_scale_parameter(bits, layer_name)
     zero_point = self._get_zero_point(bits, layer_name)
     
     # Apply dequantization: q = (r - z) * scale
     dequantized = (tensor.astype(np.float32) - zero_point) * scale
     return dequantized

    def _blockwise_dequantize(self, tensor: np.ndarray, bits: int, layer_name: str = None) -> np.ndarray:
     """Block-wise dequantization where each block has its own scale/zero-point"""
     block_size = self.quantization_info.get("block_size", 32)
     block_dim = self.quantization_info.get("block_dim", -1)
     scales = self._get_scales_for_blockwise(tensor.shape, block_size, block_dim, layer_name)
     zero_points = self._get_zero_points_for_blockwise(tensor.shape, block_size, block_dim, layer_name)
     
     # Create output tensor
     output = np.zeros(tensor.shape, dtype=np.float32)
     
     # Apply blockwise dequantization
     if block_dim == -1:  # Apply to flattened tensor
      flat_tensor = tensor.reshape(-1)
      flat_output = output.reshape(-1)
      
      for block_idx in range(0, len(flat_tensor), block_size):
       end_idx = min(block_idx + block_size, len(flat_tensor))
       scale_idx = block_idx // block_size
       scale = scales[min(scale_idx, len(scales)-1)]
       zp = zero_points[min(scale_idx, len(zero_points)-1)]
       flat_output[block_idx:end_idx] = (flat_tensor[block_idx:end_idx].astype(np.float32) - zp) * scale
       
      return output
     else:
      # Handle per-dimension blocking (more complex case)
      # This is a simplified version; real implementation would be dimension-specific
      for idx in np.ndindex(tensor.shape[:block_dim] + tensor.shape[block_dim+1:]):
       idx_prefix = idx[:block_dim]
       idx_suffix = idx[block_dim:]
       for block_start in range(0, tensor.shape[block_dim], block_size):
        block_end = min(block_start + block_size, tensor.shape[block_dim])
        scale_idx = block_start // block_size
        scale = scales[min(scale_idx, len(scales)-1)]
        zp = zero_points[min(scale_idx, len(zero_points)-1)]
        
        # Construct slices for this block
        slices = idx_prefix + (slice(block_start, block_end),) + idx_suffix
        output[slices] = (tensor[slices].astype(np.float32) - zp) * scale
      
      return output

    def _grouped_dequantize(self, tensor: np.ndarray, bits: int, layer_name: str = None) -> np.ndarray:
     """Group-wise quantization (each group of channels has different params)"""
     group_size = self.quantization_info.get("group_size", 64)
     scales = self._get_scales_for_groups(tensor.shape, group_size, layer_name)
     zero_points = self._get_zero_points_for_groups(tensor.shape, group_size, layer_name)
     
     # Assuming channel dimension is 0 (can be parameterized in real implementation)
     channel_dim = self.quantization_info.get("channel_dim", 0)
     output = np.zeros(tensor.shape, dtype=np.float32)
     
     for group_idx in range(0, tensor.shape[channel_dim], group_size):
      end_idx = min(group_idx + group_size, tensor.shape[channel_dim])
      scale_idx = group_idx // group_size
      scale = scales[min(scale_idx, len(scales)-1)]
      zp = zero_points[min(scale_idx, len(zero_points)-1)]
      
      # Create slice for this channel group
      slices = tuple(slice(None) if i != channel_dim else slice(group_idx, end_idx) 
            for i in range(len(tensor.shape)))
      output[slices] = (tensor[slices].astype(np.float32) - zp) * scale
     
     return output

    def _k_quants_dequantize(self, tensor: np.ndarray, bits: int, layer_name: str = None) -> np.ndarray:
     """K-quants special format used in some GGML models (simplified version)"""
     # This would be a complex implementation specific to GGML k-quants format
     # Here's a simplified placeholder
     quant_type = self.quantization_info.get("quant_type", "q4_0")
     
     if quant_type.startswith("q4"):
      scale_bits = 16  # For example, scales might be FP16
      dequantized = np.zeros((tensor.shape[0] * 2, tensor.shape[1] * 2), dtype=np.float32)
      # Complex unpacking logic would go here
      # For now, just return an upscaled dummy tensor with correct dimensions
      return dequantized
     else:
      # Fallback for unknown k-quants type
      return tensor.astype(np.float32)

    def _custom_dequantize(self, tensor: np.ndarray, layer_name: str = None) -> np.ndarray:
     """Custom dequantization using specified function"""
     custom_func = self.quantization_info.get("custom_dequant_func", None)
     if custom_func and callable(custom_func):
      return custom_func(tensor, self.quantization_info)
     return tensor.astype(np.float32)

    def _get_scale_parameter(self, bits: int, layer_name: str = None) -> float:
     """Get scale parameter from quantization info"""
     # Try layer-specific scale first
     if layer_name and layer_name in self.quantization_info.get("layer_specific", {}):
      if "scale" in self.quantization_info["layer_specific"][layer_name]:
       return self.quantization_info["layer_specific"][layer_name]["scale"]
     
     # Fall back to global scale or calculate from bits
     if "scale" in self.quantization_info:
      return self.quantization_info["scale"]
     else:
      # Derive scale from bit depth for symmetric quantization
      return 1.0 / (2.0 ** (bits - 1))

    def _get_zero_point(self, bits: int, layer_name: str = None) -> int:
     """Get zero point parameter from quantization info"""
     # Try layer-specific zero point first
     if layer_name and layer_name in self.quantization_info.get("layer_specific", {}):
      if "zero_point" in self.quantization_info["layer_specific"][layer_name]:
       return self.quantization_info["layer_specific"][layer_name]["zero_point"]
     
     # Fall back to global zero point or default to 0
     return self.quantization_info.get("zero_point", 0)

    def _get_scales_for_blockwise(self, shape, block_size, block_dim, layer_name: str = None) -> np.ndarray:
     """Get scales for blockwise dequantization"""
     if layer_name and layer_name in self.quantization_info.get("layer_specific", {}):
      if "block_scales" in self.quantization_info["layer_specific"][layer_name]:
       return self.quantization_info["layer_specific"][layer_name]["block_scales"]
     
     # Fall back to global block scales or generate default ones
     if "block_scales" in self.quantization_info:
      return self.quantization_info["block_scales"]
     else:
      # Calculate number of blocks and generate default scales
      if block_dim == -1:
       flat_size = np.prod(shape)
       num_blocks = (flat_size + block_size - 1) // block_size
      else:
       num_blocks = (shape[block_dim] + block_size - 1) // block_size
      return np.ones(num_blocks, dtype=np.float32) * 0.1  # Default scale

    def _get_zero_points_for_blockwise(self, shape, block_size, block_dim, layer_name: str = None) -> np.ndarray:
     """Get zero points for blockwise dequantization"""
     if layer_name and layer_name in self.quantization_info.get("layer_specific", {}):
      if "block_zero_points" in self.quantization_info["layer_specific"][layer_name]:
       return self.quantization_info["layer_specific"][layer_name]["block_zero_points"]
     
     # Fall back to global block zero points or generate default ones
     if "block_zero_points" in self.quantization_info:
      return self.quantization_info["block_zero_points"]
     else:
      # Calculate number of blocks and generate default zero points
      if block_dim == -1:
       flat_size = np.prod(shape)
       num_blocks = (flat_size + block_size - 1) // block_size
      else:
       num_blocks = (shape[block_dim] + block_size - 1) // block_size
      return np.zeros(num_blocks, dtype=np.int32)  # Default zero point

    def _get_scales_for_groups(self, shape, group_size, layer_name: str = None) -> np.ndarray:
     """Get scales for grouped dequantization"""
     if layer_name and layer_name in self.quantization_info.get("layer_specific", {}):
      if "group_scales" in self.quantization_info["layer_specific"][layer_name]:
       return self.quantization_info["layer_specific"][layer_name]["group_scales"]
     
     # Fall back to global group scales or generate default ones
     if "group_scales" in self.quantization_info:
      return self.quantization_info["group_scales"]
     else:
      # Calculate number of groups and generate default scales
      channel_dim = self.quantization_info.get("channel_dim", 0)
      num_groups = (shape[channel_dim] + group_size - 1) // group_size
      return np.ones(num_groups, dtype=np.float32) * 0.1  # Default scale

    def _get_zero_points_for_groups(self, shape, group_size, layer_name: str = None) -> np.ndarray:
     """Get zero points for grouped dequantization"""
     if layer_name and layer_name in self.quantization_info.get("layer_specific", {}):
      if "group_zero_points" in self.quantization_info["layer_specific"][layer_name]:
       return self.quantization_info["layer_specific"][layer_name]["group_zero_points"]
     
     # Fall back to global group zero points or generate default ones
     if "group_zero_points" in self.quantization_info:
      return self.quantization_info["group_zero_points"]
     else:
      # Calculate number of groups and generate default zero points
      channel_dim = self.quantization_info.get("channel_dim", 0)
      num_groups = (shape[channel_dim] + group_size - 1) // group_size
      return np.zeros(num_groups, dtype=np.int32)  # Default zero point

    def identify_artifacts(self, tensor: np.ndarray, original_tensor: np.ndarray = None, 
           confidence_threshold: float = 0.8) -> Dict[str, Any]:
     """
     Analyzes a tensor to identify quantization artifacts and estimate precision loss.
     
     This method performs comprehensive analysis including:
     - Detection of quantization scheme and its parameters (bits, scale, zero-point).
     - Statistical analysis of tensor value distribution (min, max, mean, std, histogram).
     - Identification of common quantization artifacts (saturation, value clustering, banding, zero bias).
     - Frequency domain analysis (FFT) to detect periodic patterns.
     - Calculation of quality metrics (MSE, PSNR, SQNR, SSIM) if an original tensor is provided.
     - Estimation of precision loss based on bit depth or comparison with an original tensor.
     - Generation of recommendations based on findings.
     - Calculation of an overall confidence score for the analysis.
     
     Args:
      tensor (np.ndarray): The tensor to analyze for quantization artifacts.
      original_tensor (np.ndarray, optional): Original unquantized tensor for reference
      confidence_threshold (float, optional): Threshold for confidence scores (0.0-1.0)
      
     Returns:
      Dict[str, Any]: Detailed analysis including:
       - detected_scheme: Identified quantization scheme
       - estimated_bits: Estimated bit depth
       - scale_estimate: Estimated scale parameter
       - zero_point_estimate: Estimated zero point
       - artifacts: Dict of detected artifacts with confidence scores
       - metrics: Dict of quality metrics
       - histogram_analysis: Histogram analysis results
       - confidence: Overall confidence in the analysis
     """
     results = {
      "detected_scheme": None,
      "estimated_bits": None,
      "scale_estimate": None, 
      "zero_point_estimate": None,
      "artifacts": {},
      "metrics": {},
      "histogram_analysis": {},
      "confidence": 0.0,
      "recommendations": []
     }
     
     # Skip analysis for empty tensors
     if tensor.size == 0:
      results["confidence"] = 0.0
      results["recommendations"].append("Empty tensor provided, no analysis possible")
      return results
     
     # Basic tensor statistics
     stats = self._compute_tensor_statistics(tensor)
     results["tensor_stats"] = stats
     
     # Get value distribution and histogram analysis
     hist_data = self._analyze_histogram(tensor)
     results["histogram_analysis"] = hist_data
     
     # Detect quantization scheme and parameters
     scheme_data = self._detect_quantization_scheme(tensor, hist_data)
     results.update(scheme_data)
     
     # Analyze for specific artifacts
     artifacts = self._detect_quantization_artifacts(tensor, hist_data, scheme_data)
     results["artifacts"] = artifacts
     
     # Calculate quality metrics
     metrics = self._calculate_quality_metrics(tensor, original_tensor, scheme_data)
     results["metrics"] = metrics
     
     # Perform frequency domain analysis
     if tensor.ndim >= 2:
      freq_analysis = self._frequency_domain_analysis(tensor)
      results["frequency_analysis"] = freq_analysis
      
     # Estimate precision loss
     precision_loss = self._estimate_precision_loss(tensor, original_tensor, scheme_data)
     results["precision_loss"] = precision_loss
     
     # Generate recommendations based on findings
     recommendations = self._generate_recommendations(results)
     results["recommendations"] = recommendations
     
     # Calculate overall confidence score
     confidence = self._calculate_confidence(results)
     results["confidence"] = min(confidence, 1.0)  # Cap at 1.0
     
     return results

    def _compute_tensor_statistics(self, tensor: np.ndarray) -> Dict[str, Any]:
     """Compute basic statistical properties of the tensor."""
     stats = {
      "min": float(tensor.min()),
      "max": float(tensor.max()),
      "mean": float(tensor.mean()),
      "std": float(tensor.std()),
      "median": float(np.median(tensor)),
      "unique_values": len(np.unique(tensor)),
      "shape": tensor.shape,
      "dtype": str(tensor.dtype)
     }
     
     # Check for zeros and infinities
     stats["zero_percentage"] = float(np.sum(tensor == 0) / tensor.size * 100)
     stats["inf_percentage"] = float(np.sum(~np.isfinite(tensor)) / tensor.size * 100)
     
     # Compute skewness and kurtosis for distribution analysis
     if tensor.size > 3:  # Need at least 4 values for kurtosis
      stats["skewness"] = float(scipy_stats.skew(tensor.flatten()))
      stats["kurtosis"] = float(scipy_stats.kurtosis(tensor.flatten()))
     
     return stats

    def _analyze_histogram(self, tensor: np.ndarray) -> Dict[str, Any]:
     """Analyze histogram of tensor values to identify patterns."""
     flat_tensor = tensor.flatten()
     
     # Create histogram with adaptive bin count
     bin_count = min(256, len(np.unique(flat_tensor)))
     bin_count = max(bin_count, 10)  # At least 10 bins
     
     hist, bin_edges = np.histogram(flat_tensor, bins=bin_count)
     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
     
     # Find peaks in histogram (potential quantization levels)
     if len(hist) > 3:
      peak_indices, peak_props = signal.find_peaks(hist, height=np.max(hist)/10)
      peaks = bin_centers[peak_indices]
      peak_heights = peak_props['peak_heights']
     else:
      peaks = []
      peak_heights = []
     
     # Detect regular spacing in peaks (sign of uniform quantization)
     peak_spacing = []
     if len(peaks) >= 2:
      peak_spacing = np.diff(peaks)
     
     # Detect gaps in the histogram (empty bins between quantization levels)
     gaps = []
     if len(hist) > 0:
      gap_threshold = np.max(hist) * 0.01
      gap_indices = np.where(hist < gap_threshold)[0]
      if len(gap_indices) > 0:
       gaps = bin_centers[gap_indices]
     
     return {
      "histogram": hist.tolist(),
      "bin_edges": bin_edges.tolist(),
      "peaks": {
       "positions": peaks.tolist() if len(peaks) > 0 else [],
       "heights": peak_heights.tolist() if len(peak_heights) > 0 else [],
       "count": len(peaks)
      },
      "peak_spacing_stats": {
       "mean": float(np.mean(peak_spacing)) if len(peak_spacing) > 0 else 0,
       "std": float(np.std(peak_spacing)) if len(peak_spacing) > 0 else 0,
       "values": peak_spacing.tolist() if len(peak_spacing) > 0 else []
      },
      "gaps": {
       "count": len(gaps),
       "positions": gaps.tolist() if len(gaps) > 0 else []
      }
     }

    def _detect_quantization_scheme(self, tensor: np.ndarray, hist_data: Dict[str, Any]) -> Dict[str, Any]:
     """Detect quantization scheme based on tensor values and histogram analysis."""
     unique_values = np.unique(tensor)
     num_unique = len(unique_values)
     
     # Estimate bit depth
     if num_unique <= 1:
      estimated_bits = 0  # Constant tensor
     else:
      estimated_bits = max(1, np.ceil(np.log2(num_unique))).astype(int)
      
      # Check if values cluster around powers of 2
      if estimated_bits <= 16:
       # Check if values match expected pattern for common bit depths
       for bits in [1, 2, 4, 8, 16]:
        if bits >= estimated_bits:
         if self._check_bit_pattern_match(tensor, bits):
          estimated_bits = bits
          break
     
     # Detect if symmetric or asymmetric
     scheme_type = "unknown"
     zero_point = 0
     scale = 1.0
     
     # For symmetric schemes, values are typically centered around 0
     if np.abs(np.abs(tensor.min()) - tensor.max()) / (max(np.abs(tensor.max()), np.abs(tensor.min())) + 1e-10) < 0.2:
      scheme_type = "symmetric_uniform"
      max_abs = max(np.abs(tensor.min()), np.abs(tensor.max()))
      if max_abs > 0:
       scale = max_abs / (2**(estimated_bits-1) - 1)
     else:
      scheme_type = "asymmetric_uniform"
      # Estimate zero point and scale
      range_values = tensor.max() - tensor.min()
      if range_values > 0:
       scale = range_values / (2**estimated_bits - 1)
       zero_point = -int(round(tensor.min() / scale))
     
     # Check for group-wise or blockwise schemes
     has_blockwise = False
     has_groupwise = False
     
     # If tensor is 2D or more, check variance along different dimensions
     if tensor.ndim >= 2:
      # Check blockwise pattern by analyzing variance in blocks
      block_size = min(32, min(tensor.shape))
      if block_size > 1:
       has_blockwise = self._check_for_blockwise_pattern(tensor, block_size)
      
      # Check groupwise pattern by analyzing channel statistics
      if tensor.shape[0] > 8:  # Only check if enough channels
       has_groupwise = self._check_for_groupwise_pattern(tensor)
     
     # Determine confidence in scheme detection
     confidence = 0.7  # Base confidence
     
     # Adjust confidence based on findings
     if num_unique < 5:
      confidence *= 0.6  # Low unique values increases uncertainty
     if hist_data["peaks"]["count"] > 1:
      spacing_std = hist_data["peak_spacing_stats"]["std"]
      spacing_mean = hist_data["peak_spacing_stats"]["mean"]
      if spacing_mean > 0 and spacing_std / spacing_mean < 0.1:
       confidence *= 1.2  # Regular peak spacing increases confidence
     
     return {
      "detected_scheme": scheme_type,
      "estimated_bits": int(estimated_bits),
      "scale_estimate": float(scale),
      "zero_point_estimate": int(zero_point),
      "has_blockwise_pattern": bool(has_blockwise),
      "has_groupwise_pattern": bool(has_groupwise),
      "unique_value_count": int(num_unique),
      "scheme_detection_confidence": min(float(confidence), 1.0)
     }

    def _check_bit_pattern_match(self, tensor: np.ndarray, bits: int) -> bool:
     """Check if tensor values match expected pattern for specified bit depth."""
     unique_values = np.unique(tensor)
     
     # For symmetric quantization, expect values from -2^(bits-1) to 2^(bits-1)-1
     expected_symmetric = set(range(-(2**(bits-1)), 2**(bits-1)))
     
     # For asymmetric quantization, expect values from 0 to 2^bits-1
     expected_asymmetric = set(range(0, 2**bits))
     
     # Calculate overlap with expected patterns
     symmetric_overlap = len(set(unique_values.astype(int)) & expected_symmetric) / len(unique_values)
     asymmetric_overlap = len(set(unique_values.astype(int)) & expected_asymmetric) / len(unique_values)
     
     return max(symmetric_overlap, asymmetric_overlap) > 0.9

    def _check_for_blockwise_pattern(self, tensor: np.ndarray, block_size: int) -> bool:
     """Detect if tensor appears to use blockwise quantization."""
     # Simple heuristic: check variance of statistics across blocks
     blocks = []
     for i in range(0, tensor.shape[0], block_size):
      for j in range(0, tensor.shape[1], block_size):
       block = tensor[i:min(i+block_size, tensor.shape[0]), 
            j:min(j+block_size, tensor.shape[1])]
       blocks.append((block.min(), block.max(), block.mean()))
     
     blocks = np.array(blocks)
     if len(blocks) < 2:
      return False
     
     # Calculate coefficient of variation for ranges within blocks
     ranges = blocks[:, 1] - blocks[:, 0]
     mean_range = np.mean(ranges)
     if mean_range == 0:
      return False
     
     cv_ranges = np.std(ranges) / mean_range
     
     # High variation in block ranges suggests blockwise quantization
     return cv_ranges > 0.3

    def _check_for_groupwise_pattern(self, tensor: np.ndarray) -> bool:
     """Detect if tensor appears to use groupwise quantization."""
     # Assumes channel dimension is first dimension
     if tensor.shape[0] < 8:  # Need enough channels to detect pattern
      return False
     
     # Check statistics per channel
     channel_stats = []
     for i in range(tensor.shape[0]):
      channel = tensor[i]
      channel_stats.append((channel.min(), channel.max(), channel.mean(), channel.std()))
     
     channel_stats = np.array(channel_stats)
     
     # Calculate variance of statistics across channels
     range_variance = np.var(channel_stats[:, 1] - channel_stats[:, 0])
     mean_variance = np.var(channel_stats[:, 2])
     
     # Check for patterns by group (e.g., 4, 8, 16, 32 channels per group)
     group_sizes = [4, 8, 16, 32]
     max_pattern_score = 0
     
     for g in group_sizes:
      if tensor.shape[0] < g*2:  # Need at least 2 groups
       continue
       
      # Compare intra-group vs inter-group variance
      intra_group_variance = 0
      inter_group_variance = 0
      
      group_means = []
      for i in range(0, tensor.shape[0], g):
       group = channel_stats[i:i+g]
       if len(group) > 0:
        group_mean = np.mean(group[:, 2])  # Mean of channel means
        group_means.append(group_mean)
        intra_group_variance += np.var(group[:, 2])
      
      if len(group_means) > 1:
       inter_group_variance = np.var(group_means)
       
      if intra_group_variance > 0:
       pattern_score = inter_group_variance / (intra_group_variance + 1e-10)
       max_pattern_score = max(max_pattern_score, pattern_score)
     
     return max_pattern_score > 1.5  # Threshold determined empirically

    def _detect_quantization_artifacts(self, tensor: np.ndarray, 
             hist_data: Dict[str, Any], 
             scheme_data: Dict[str, Any]) -> Dict[str, Any]:
     """Detect various quantization artifacts in the tensor."""
     artifacts = {}
     
     # Check for clipping/saturation at extremes
     min_val, max_val = tensor.min(), tensor.max()
     hist = hist_data["histogram"]
     bin_edges = hist_data["bin_edges"]
     
     # Detect saturation at min/max values
     min_saturation = float(np.sum(tensor == min_val) / tensor.size)
     max_saturation = float(np.sum(tensor == max_val) / tensor.size)
     
     artifacts["saturation"] = {
      "min_saturation_percent": min_saturation * 100,
      "max_saturation_percent": max_saturation * 100,
      "has_significant_saturation": bool(min_saturation > 0.01 or max_saturation > 0.01),
      "confidence": float(0.9 if min_saturation > 0.01 or max_saturation > 0.01 else 0.5)
     }
     
     # Detect value clustering (values cluster around quantization levels)
     if len(hist_data["peaks"]["positions"]) > 0:
      peak_count = len(hist_data["peaks"]["positions"])
      expected_peaks = min(2**scheme_data["estimated_bits"], 50)
      artifacts["value_clustering"] = {
       "peak_count": peak_count,
       "clustering_strength": float(np.max(hist) / np.mean(hist) if np.mean(hist) > 0 else 0),
       "has_significant_clustering": bool(peak_count > 2 and peak_count < len(hist) / 2),
       "confidence": float(0.8 if peak_count > 0 else 0.4)
      }
     
     # Detect banding patterns (visible steps in what should be smooth transitions)
     banding_score = 0.0
     if len(hist) > 10:
      # Calculate derivative of histogram to find sharp transitions
      hist_derivative = np.abs(np.diff(hist))
      banding_score = np.max(hist_derivative) / (np.mean(hist) + 1e-10)
     
     artifacts["banding"] = {
      "banding_score": float(banding_score),
      "has_banding": bool(banding_score > 5.0),
      "confidence": float(min(0.9, banding_score / 10))
     }
     
     # Detect bias/shift from zero
     if 0 in np.unique(tensor):
      zero_bin_idx = np.argmin(np.abs(bin_edges[:-1] - 0))
      zero_bin_height = hist[zero_bin_idx] if zero_bin_idx < len(hist) else 0
      expected_zero_height = np.mean(hist)
      zero_bias = (zero_bin_height - expected_zero_height) / (expected_zero_height + 1e-10)
      
      artifacts["zero_bias"] = {
       "bias_score": float(zero_bias),
       "has_zero_bias": bool(abs(zero_bias) > 0.5),
       "confidence": float(min(0.8, abs(zero_bias) / 2))
      }
     
     return artifacts

    def _calculate_quality_metrics(self, tensor: np.ndarray, 
            original_tensor: np.ndarray = None,
            scheme_data: Dict[str, Any] = None) -> Dict[str, Any]:
     """Calculate quality metrics for the quantized tensor."""
     metrics = {}
     
     # If original tensor is available, calculate direct comparison metrics
     if original_tensor is not None and tensor.shape == original_tensor.shape:
      # Calculate Mean Squared Error
      # "expected_peaks": expected_pe # Typo removed
      mse = np.mean((tensor - original_tensor) ** 2)
      metrics["mse"] = float(mse)
      
      # Calculate Peak Signal-to-Noise Ratio
      data_range = original_tensor.max() - original_tensor.min()
      if mse > 0 and data_range > 0:
       psnr = 20 * np.log10(data_range / np.sqrt(mse))
       metrics["psnr"] = float(psnr)
      else:
       metrics["psnr"] = float('inf')
      
      # Calculate Signal-to-Quantization-Noise Ratio
      signal_power = np.mean(original_tensor ** 2)
      noise_power = mse
      if noise_power > 0 and signal_power > 0:
       sqnr = 10 * np.log10(signal_power / noise_power)
       metrics["sqnr"] = float(sqnr)
      else:
       metrics["sqnr"] = float('inf')
       
      # Try to calculate SSIM if scipy is available
      try:
       from scipy.ndimage import gaussian_filter
       
       def ssim(x, y, data_range):
        # Constants for stability
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        
        # Compute means
        ux = gaussian_filter(x, sigma=1.5)
        uy = gaussian_filter(y, sigma=1.5)
        
        # Compute variances and covariance
        uxx = gaussian_filter(x * x, sigma=1.5)
        uyy = gaussian_filter(y * y, sigma=1.5)
        uxy = gaussian_filter(x * y, sigma=1.5)
        
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy
        
        # Calculate SSIM
        num = (2 * ux * uy + c1) * (2 * vxy + c2)
        den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
        return num / den
       
       # Calculate SSIM for the tensor
       if tensor.ndim >= 2:
        ssim_value = np.mean(ssim(original_tensor, tensor, data_range))
        metrics["ssim"] = float(ssim_value)
        
      except Exception as e:
       metrics["ssim_error"] = str(e)
     return metrics

    def _frequency_domain_analysis(self, tensor: np.ndarray) -> Dict[str, Any]:
     """
     Performs basic frequency domain analysis using FFT on the flattened tensor.
     This can help identify periodic artifacts or strong regular patterns that might
     be related to quantization, especially banding or interference.

     Args:
        tensor (np.ndarray): The input tensor.

     Returns:
        Dict[str, Any]: A dictionary containing:
            - 'dominant_frequencies': List of top 5 dominant frequencies (excluding DC).
            - 'dominant_powers': Corresponding power spectrum values for these frequencies.
            - 'has_strong_periodic_signal': Boolean indicating if a strong periodic signal
                                            was heuristically detected.
            - 'fft_applied_to_flattened': True, indicating the analysis was on the flattened tensor.
            - 'error': Error message if FFT analysis failed or tensor was unsuitable.
     """
     if tensor.ndim == 0 or tensor.size < 2: # Cannot perform FFT on scalar or single value
        return {"error": "Tensor not suitable for FFT (scalar or too small).", "dominant_frequencies": [], "dominant_powers": [], "has_strong_periodic_signal": False}

     analysis = {}
     try:
        # For multi-dimensional tensors, analyze 1D slices or aggregate
        # Here, we take the FFT of the flattened tensor for simplicity
        flat_tensor = tensor.flatten().astype(np.float32)

        # Perform FFT
        fft_result = np.fft.fft(flat_tensor)
        fft_freq = np.fft.fftfreq(flat_tensor.size)

        # Get power spectrum (magnitude squared)
        power_spectrum = np.abs(fft_result)**2

        # Identify dominant frequencies (excluding DC component at index 0)
        if flat_tensor.size > 1:
            dominant_freq_indices = np.argsort(power_spectrum[1:])[-5:] + 1 # Top 5 excluding DC
            dominant_frequencies = np.abs(fft_freq[dominant_freq_indices])
            dominant_powers = power_spectrum[dominant_freq_indices]

            analysis["dominant_frequencies"] = dominant_frequencies.tolist()
            analysis["dominant_powers"] = dominant_powers.tolist()
            analysis["has_strong_periodic_signal"] = bool(np.max(dominant_powers) > (np.mean(power_spectrum[1:]) * 10)) # Heuristic
        else:
            analysis["dominant_frequencies"] = []
            analysis["dominant_powers"] = []
            analysis["has_strong_periodic_signal"] = False

        analysis["fft_applied_to_flattened"] = True

     except Exception as e:
        analysis["error"] = f"FFT analysis failed: {str(e)}"
     return analysis

    def _estimate_precision_loss(self, tensor: np.ndarray,
                               original_tensor: np.ndarray = None,
                               scheme_data: Dict[str, Any] = None) -> Dict[str, Any]:
     """
     Estimates precision loss due to quantization.

     If an original (unquantized) tensor is provided, loss is estimated by direct
     comparison (e.g. difference statistics, reduction in unique values).
     Otherwise, if scheme data (like estimated bit depth) is available, loss is
     estimated based on the reduction from an assumed original precision (e.g., 32-bit float).

     Args:
        tensor (np.ndarray): The quantized tensor.
        original_tensor (np.ndarray, optional): The original unquantized tensor.
        scheme_data (Dict[str, Any], optional): Data about the detected quantization scheme,
                                                including 'estimated_bits'.

     Returns:
        Dict[str, Any]: A dictionary containing:
            - 'method': The method used for estimation ('direct_comparison',
                        'bit_depth_estimation', or 'unknown').
            - 'details': A sub-dictionary with specific loss metrics, e.g.,
                         'diff_mean_abs', 'diff_std', 'max_abs_error' (for direct comparison),
                         or 'estimated_bit_reduction', 'quantization_levels_lost_factor'
                         (for bit depth estimation).
     """
     loss_info = {"method": None, "details": {}}

     if original_tensor is not None and tensor.shape == original_tensor.shape:
        loss_info["method"] = "direct_comparison"
        # Calculate difference statistics
        diff = original_tensor.astype(np.float64) - tensor.astype(np.float64) # Use float64 for precision
        loss_info["details"]["diff_mean_abs"] = float(np.mean(np.abs(diff)))
        loss_info["details"]["diff_std"] = float(np.std(diff))
        loss_info["details"]["max_abs_error"] = float(np.max(np.abs(diff)))

        # Simplified entropy change (requires careful binning for true information loss)
        # For now, let's use unique value count as a proxy if bits are not well estimated
        original_unique = len(np.unique(original_tensor))
        quantized_unique = len(np.unique(tensor))
        loss_info["details"]["unique_value_reduction_ratio"] = float(quantized_unique / original_unique) if original_unique > 0 else 1.0

     elif scheme_data and "estimated_bits" in scheme_data:
        loss_info["method"] = "bit_depth_estimation"
        estimated_bits = scheme_data["estimated_bits"]
        # Assuming original was high precision (e.g., 32-bit float)
        original_bits = 32
        if estimated_bits > 0 and estimated_bits < original_bits :
            loss_info["details"]["estimated_bit_reduction"] = original_bits - estimated_bits
            loss_info["details"]["quantization_levels_lost_factor"] = float(2**(original_bits - estimated_bits))
        else:
            loss_info["details"]["estimated_bit_reduction"] = 0
            loss_info["details"]["quantization_levels_lost_factor"] = 1.0

     else:
        loss_info["method"] = "unknown"
        loss_info["details"]["message"] = "Insufficient data to estimate precision loss."

     return loss_info

    def _generate_recommendations(self, results: Dict[str, Any]) -> list[str]:
     """
     Generates a list of textual recommendations based on the analysis results.
     These recommendations highlight potential issues or areas for further investigation.

     Args:
        results (Dict[str, Any]): The comprehensive analysis results dictionary
                                  from `identify_artifacts`.

     Returns:
        list[str]: A list of string recommendations.
     """
     recommendations = []

     # Scheme detection confidence
     if results.get("scheme_detection_confidence", 1.0) < 0.6:
        recommendations.append("Quantization scheme detection confidence is low. Results may be less reliable.")

     # Artifacts
     artifacts = results.get("artifacts", {})
     if artifacts.get("saturation", {}).get("has_significant_saturation"):
        recommendations.append("Significant saturation detected at min/max values. Consider adjusting quantization range or clipping values.")
     if artifacts.get("banding", {}).get("has_banding"):
        recommendations.append("Banding artifacts detected. This may indicate overly aggressive quantization or issues with value distribution.")
     if artifacts.get("value_clustering", {}).get("has_significant_clustering"):
        if artifacts["value_clustering"]["peak_count"] < 2**results.get("estimated_bits", 8) / 2: # Heuristic
            recommendations.append("Value clustering suggests effective bit depth might be lower than estimated or data has sparse value representation.")
     if artifacts.get("zero_bias",{}).get("has_zero_bias"):
        recommendations.append("Bias detected around zero. Check if this is expected or an artifact.")

     # Quality Metrics (if original tensor was provided)
     metrics = results.get("metrics", {})
     if "psnr" in metrics and metrics["psnr"] != float('inf') and metrics["psnr"] < 20: # Low PSNR
        recommendations.append(f"Low PSNR ({metrics['psnr']:.2f} dB) indicates significant quality loss compared to original.")
     if "ssim" in metrics and metrics["ssim"] < 0.8: # Low SSIM
        recommendations.append(f"Low SSIM ({metrics['ssim']:.3f}) indicates structural dissimilarity from original.")

     # Precision Loss
     precision_loss = results.get("precision_loss", {})
     if precision_loss.get("details", {}).get("estimated_bit_reduction", 0) > 16:
        recommendations.append("High estimated bit reduction. Significant precision loss is likely.")
     if precision_loss.get("details", {}).get("max_abs_error", 0) > (results.get("tensor_stats",{}).get("max",1) - results.get("tensor_stats",{}).get("min",0)) * 0.1: # Large max error
         recommendations.append("Maximum absolute error is high relative to tensor range.")

     if not recommendations:
        recommendations.append("No major issues detected based on current heuristics.")

     return recommendations

    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
     """
     Calculates an overall confidence score for the quantization artifact analysis.

     This score is a heuristic aggregation of various confidence factors derived
     during the analysis, such as scheme detection confidence, artifact detection
     confidence, consistency of findings, and quality metrics if available.

     Args:
        results (Dict[str, Any]): The comprehensive analysis results dictionary
                                  from `identify_artifacts`.

     Returns:
        float: An overall confidence score between 0.0 and 1.0.
     """
     total_confidence_score = 0
     num_factors = 0

     # Scheme detection confidence
     scheme_confidence = results.get("scheme_detection_confidence", 0.5)
     total_confidence_score += scheme_confidence
     num_factors += 1

     # Artifact confidence (average confidence of detected artifacts)
     artifact_confidences = []
     for artifact_type, artifact_data in results.get("artifacts", {}).items():
        if isinstance(artifact_data, dict) and "confidence" in artifact_data:
            artifact_confidences.append(artifact_data["confidence"])

     if artifact_confidences:
        avg_artifact_confidence = np.mean(artifact_confidences)
        total_confidence_score += avg_artifact_confidence
        num_factors += 1
     else: # No specific artifacts with confidence, use a neutral value
        total_confidence_score += 0.7 # Base confidence if no specific artifacts are flagged
        num_factors += 1

     # Consistency between estimated bits and unique values
     estimated_bits = results.get("estimated_bits")
     unique_values = results.get("tensor_stats", {}).get("unique_values")
     if estimated_bits is not None and unique_values is not None and unique_values > 1:
        expected_max_unique = 2**estimated_bits
        # Confidence drops if unique values are far from expected for estimated_bits
        if unique_values > expected_max_unique or unique_values < expected_max_unique / 4 : # Heuristic range
            consistency_confidence = 0.6
        else:
            consistency_confidence = 0.9
        total_confidence_score += consistency_confidence
        num_factors +=1

     # If original tensor provided, metrics quality can influence confidence
     if "mse" in results.get("metrics", {}):
        # Example: if PSNR is very high, confidence in analysis might be higher
        psnr = results["metrics"].get("psnr", 0)
        if psnr == float('inf') or psnr > 40: # Very good quality
            total_confidence_score += 0.95
        elif psnr > 25: # Good quality
            total_confidence_score += 0.8
        else: # Lower quality
            total_confidence_score += 0.65
        num_factors += 1

     if num_factors == 0:
        return 0.5 # Default if no factors contributed

     final_confidence = total_confidence_score / num_factors
     return min(max(final_confidence, 0.0), 1.0) # Ensure it's between 0 and 1
