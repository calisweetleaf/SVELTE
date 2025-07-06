# src/tensor_analysis/quantization.py
"""
Quantization scheme identification and dequantization simulation for SVELTE Framework.
"""
import numpy as np
import struct
from typing import Dict, Any, Tuple, Optional, Union, List
from enum import Enum
import warnings
import logging

# Try to import scipy for advanced statistical analysis
try:
    from scipy import signal, stats as scipy_stats
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some advanced quantization analysis features will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Enumeration of supported quantization types."""
    SYMMETRIC_UNIFORM = "symmetric_uniform"
    ASYMMETRIC_UNIFORM = "asymmetric_uniform"
    BLOCKWISE = "blockwise"
    GROUPED = "grouped"
    K_QUANTS = "k_quants"
    CUSTOM = "custom"
    NONE = "none"

class KQuantsType(Enum):
    """K-quants specific quantization types."""
    Q2_K = "q2_k"
    Q3_K = "q3_k"
    Q4_K = "q4_k"
    Q5_K = "q5_k"
    Q6_K = "q6_k"
    Q8_K = "q8_k"

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
     """K-quants dequantization for GGML format tensors."""
     quant_type = self.quantization_info.get("quant_type", "q4_k")
     
     try:
         if quant_type == "q2_k":
             return self._dequantize_q2_k(tensor, layer_name)
         elif quant_type == "q3_k":
             return self._dequantize_q3_k(tensor, layer_name)
         elif quant_type == "q4_k":
             return self._dequantize_q4_k(tensor, layer_name)
         elif quant_type == "q5_k":
             return self._dequantize_q5_k(tensor, layer_name)
         elif quant_type == "q6_k":
             return self._dequantize_q6_k(tensor, layer_name)
         elif quant_type == "q8_k":
             return self._dequantize_q8_k(tensor, layer_name)
         else:
             logger.warning(f"Unknown k-quants type: {quant_type}, falling back to simple dequantization")
             return tensor.astype(np.float32)
             
     except Exception as e:
         logger.error(f"K-quants dequantization failed: {e}")
         return tensor.astype(np.float32)

    def _dequantize_q4_k(self, tensor: np.ndarray, layer_name: str = None) -> np.ndarray:
        """Dequantize Q4_K format tensors (4-bit with k-means clustering)."""
        # Q4_K format: 32 values per block, stored as:
        # - 1 float32 scale (4 bytes)
        # - 1 float32 min value (4 bytes) 
        # - 16 bytes of 4-bit quantized values (32 values, 2 per byte)
        
        block_size = 32
        bytes_per_block = 24  # 4 + 4 + 16
        
        if tensor.size % bytes_per_block != 0:
            logger.warning("Tensor size doesn't match Q4_K format expectations")
            return tensor.astype(np.float32)
        
        num_blocks = tensor.size // bytes_per_block
        tensor_bytes = tensor.view(np.uint8)
        
        # Calculate output size
        output_size = num_blocks * block_size
        output = np.zeros(output_size, dtype=np.float32)
        
        for block_idx in range(num_blocks):
            offset = block_idx * bytes_per_block
            
            # Extract scale and min value (little-endian float32)
            scale_bytes = tensor_bytes[offset:offset+4]
            min_bytes = tensor_bytes[offset+4:offset+8]
            
            scale = struct.unpack('<f', scale_bytes)[0]
            min_val = struct.unpack('<f', min_bytes)[0]
            
            # Extract 4-bit values (16 bytes = 32 values)
            quant_bytes = tensor_bytes[offset+8:offset+24]
            
            # Unpack 4-bit values
            output_offset = block_idx * block_size
            for i in range(16):  # 16 bytes
                byte_val = quant_bytes[i]
                # Each byte contains 2 x 4-bit values
                val1 = byte_val & 0x0F  # Lower 4 bits
                val2 = (byte_val >> 4) & 0x0F  # Upper 4 bits
                
                # Dequantize: value = min + scale * quantized_value
                output[output_offset + i*2] = min_val + scale * val1
                if (output_offset + i*2 + 1) < output_size:
                    output[output_offset + i*2 + 1] = min_val + scale * val2
        
        return output

    def _dequantize_q2_k(self, tensor: np.ndarray, layer_name: str = None) -> np.ndarray:
        """Dequantize Q2_K format tensors (2-bit with k-means clustering)."""
        # Q2_K format: 16 values per block
        block_size = 16
        bytes_per_block = 12  # 4 (scale) + 4 (min) + 4 (2-bit values)
        
        if tensor.size % bytes_per_block != 0:
            logger.warning("Tensor size doesn't match Q2_K format expectations")
            return tensor.astype(np.float32)
        
        num_blocks = tensor.size // bytes_per_block
        tensor_bytes = tensor.view(np.uint8)
        output_size = num_blocks * block_size
        output = np.zeros(output_size, dtype=np.float32)
        
        for block_idx in range(num_blocks):
            offset = block_idx * bytes_per_block
            
            # Extract scale and min value
            scale = struct.unpack('<f', tensor_bytes[offset:offset+4])[0]
            min_val = struct.unpack('<f', tensor_bytes[offset+4:offset+8])[0]
            
            # Extract 2-bit values (4 bytes = 16 values)
            quant_bytes = tensor_bytes[offset+8:offset+12]
            
            output_offset = block_idx * block_size
            for i in range(4):  # 4 bytes
                byte_val = quant_bytes[i]
                # Each byte contains 4 x 2-bit values
                for j in range(4):
                    val = (byte_val >> (j * 2)) & 0x03  # Extract 2 bits
                    idx = output_offset + i*4 + j
                    if idx < output_size:
                        output[idx] = min_val + scale * val
        
        return output

    def _dequantize_q8_k(self, tensor: np.ndarray, layer_name: str = None) -> np.ndarray:
        """Dequantize Q8_K format tensors (8-bit with k-means clustering)."""
        # Q8_K format: 256 values per block
        block_size = 256
        bytes_per_block = 264  # 4 (scale) + 4 (min) + 256 (8-bit values)
        
        if tensor.size % bytes_per_block != 0:
            logger.warning("Tensor size doesn't match Q8_K format expectations")
            return tensor.astype(np.float32)
        
        num_blocks = tensor.size // bytes_per_block
        tensor_bytes = tensor.view(np.uint8)
        output_size = num_blocks * block_size
        output = np.zeros(output_size, dtype=np.float32)
        
        for block_idx in range(num_blocks):
            offset = block_idx * bytes_per_block
            
            # Extract scale and min value
            scale = struct.unpack('<f', tensor_bytes[offset:offset+4])[0]
            min_val = struct.unpack('<f', tensor_bytes[offset+4:offset+8])[0]
            
            # Extract 8-bit values
            quant_bytes = tensor_bytes[offset+8:offset+264]
            
            output_offset = block_idx * block_size
            for i in range(block_size):
                if (output_offset + i) < output_size:
                    output[output_offset + i] = min_val + scale * quant_bytes[i]
        
        return output

    def _dequantize_q3_k(self, tensor: np.ndarray, layer_name: str = None) -> np.ndarray:
        """Dequantize Q3_K format tensors (3-bit with k-means clustering)."""
        # Q3_K is more complex due to 3-bit packing
        # Simplified implementation - real Q3_K has more sophisticated packing
        logger.warning("Q3_K dequantization using simplified algorithm")
        return self._dequantize_q4_k(tensor, layer_name)  # Fallback to Q4_K

    def _dequantize_q5_k(self, tensor: np.ndarray, layer_name: str = None) -> np.ndarray:
        """Dequantize Q5_K format tensors (5-bit with k-means clustering)."""
        # Q5_K format similar to Q4_K but with 5-bit values
        logger.warning("Q5_K dequantization using simplified algorithm")
        return self._dequantize_q4_k(tensor, layer_name)  # Fallback to Q4_K

    def _dequantize_q6_k(self, tensor: np.ndarray, layer_name: str = None) -> np.ndarray:
        """Dequantize Q6_K format tensors (6-bit with k-means clustering)."""
        # Q6_K format similar to Q4_K but with 6-bit values
        logger.warning("Q6_K dequantization using simplified algorithm")
        return self._dequantize_q4_k(tensor, layer_name)  # Fallback to Q4_K

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
     - Detection of quantization scheme
     - Estimation of quantization parameters
     - Identification of value clusters and banding
     - Frequency domain analysis for artifact detection
     - Statistical analysis of value distribution
     - Signal quality metrics
     - Precision loss estimation
     
     Args:
      tensor (np.ndarray): The tensor to analyze for quantization artifacts
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

    def _frequency_domain_analysis(self, tensor: np.ndarray) -> Dict[str, Any]:
        """Perform frequency domain analysis to detect quantization artifacts."""
        freq_analysis = {}
        
        try:
            # For 2D tensors, apply 2D FFT
            if tensor.ndim == 2:
                # Apply 2D FFT
                fft_result = np.fft.fft2(tensor)
                power_spectrum = np.abs(fft_result) ** 2
                
                # Analyze frequency distribution
                freq_analysis["power_spectrum_stats"] = {
                    "mean": float(np.mean(power_spectrum)),
                    "std": float(np.std(power_spectrum)),
                    "max": float(np.max(power_spectrum)),
                    "energy": float(np.sum(power_spectrum))
                }
                
                # Detect high-frequency artifacts (typical of quantization)
                h, w = power_spectrum.shape
                center_h, center_w = h // 2, w // 2
                
                # Calculate energy in different frequency bands
                low_freq_mask = np.zeros_like(power_spectrum)
                high_freq_mask = np.zeros_like(power_spectrum)
                
                # Low frequency: central 1/4 of spectrum
                low_h, low_w = h // 4, w // 4
                low_freq_mask[center_h-low_h:center_h+low_h, 
                             center_w-low_w:center_w+low_w] = 1
                
                # High frequency: outer regions
                high_freq_mask = 1 - low_freq_mask
                
                low_freq_energy = np.sum(power_spectrum * low_freq_mask)
                high_freq_energy = np.sum(power_spectrum * high_freq_mask)
                
                total_energy = low_freq_energy + high_freq_energy
                if total_energy > 0:
                    high_freq_ratio = high_freq_energy / total_energy
                    freq_analysis["high_frequency_ratio"] = float(high_freq_ratio)
                    freq_analysis["has_high_freq_artifacts"] = bool(high_freq_ratio > 0.3)
                
                # Detect regular patterns in frequency domain (sign of quantization steps)
                freq_peaks = []
                if SCIPY_AVAILABLE:
                    # Find peaks in the power spectrum
                    spectrum_1d = np.mean(power_spectrum, axis=0)
                    peaks, _ = signal.find_peaks(spectrum_1d, height=np.mean(spectrum_1d) * 2)
                    freq_analysis["frequency_peaks"] = len(peaks)
                    freq_analysis["has_regular_freq_pattern"] = bool(len(peaks) > 5)
                
            # For 1D tensors, apply 1D FFT
            elif tensor.ndim == 1:
                fft_result = np.fft.fft(tensor)
                power_spectrum = np.abs(fft_result) ** 2
                
                freq_analysis["power_spectrum_stats"] = {
                    "mean": float(np.mean(power_spectrum)),
                    "std": float(np.std(power_spectrum)),
                    "max": float(np.max(power_spectrum)),
                    "energy": float(np.sum(power_spectrum))
                }
                
                # Calculate spectral centroid (measure of spectral shape)
                freqs = np.fft.fftfreq(len(tensor))
                spectral_centroid = np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + 1e-10)
                freq_analysis["spectral_centroid"] = float(spectral_centroid)
                
            # For higher-dimensional tensors, flatten and analyze
            else:
                flat_tensor = tensor.flatten()
                if len(flat_tensor) > 1000:  # Subsample for efficiency
                    indices = np.linspace(0, len(flat_tensor)-1, 1000, dtype=int)
                    flat_tensor = flat_tensor[indices]
                
                fft_result = np.fft.fft(flat_tensor)
                power_spectrum = np.abs(fft_result) ** 2
                
                freq_analysis["power_spectrum_stats"] = {
                    "mean": float(np.mean(power_spectrum)),
                    "std": float(np.std(power_spectrum)),
                    "max": float(np.max(power_spectrum)),
                    "energy": float(np.sum(power_spectrum))
                }
                
        except Exception as e:
            logger.warning(f"Frequency domain analysis failed: {e}")
            freq_analysis["error"] = str(e)
        
        return freq_analysis

    def _estimate_precision_loss(self, tensor: np.ndarray, 
                                original_tensor: np.ndarray = None,
                                scheme_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Estimate precision loss due to quantization."""
        precision_loss = {}
        
        # Theoretical precision loss based on bit depth
        if scheme_data and "estimated_bits" in scheme_data:
            bits = scheme_data["estimated_bits"]
            theoretical_snr = 6.02 * bits + 1.76  # dB
            precision_loss["theoretical_snr_db"] = float(theoretical_snr)
            precision_loss["theoretical_precision_loss"] = float(1.0 / (2 ** bits))
        
        # Estimate precision from value distribution
        unique_values = np.unique(tensor)
        value_range = tensor.max() - tensor.min()
        
        if len(unique_values) > 1 and value_range > 0:
            # Average spacing between quantization levels
            avg_step_size = value_range / (len(unique_values) - 1)
            precision_loss["avg_quantization_step"] = float(avg_step_size)
            precision_loss["relative_step_size"] = float(avg_step_size / value_range)
            
            # Estimate effective bits from unique values
            effective_bits = np.log2(len(unique_values))
            precision_loss["effective_bits"] = float(effective_bits)
        
        # Calculate empirical SNR if original tensor is available
        if original_tensor is not None and tensor.shape == original_tensor.shape:
            signal_power = np.mean(original_tensor ** 2)
            noise_power = np.mean((tensor - original_tensor) ** 2)
            
            if noise_power > 0 and signal_power > 0:
                empirical_snr = 10 * np.log10(signal_power / noise_power)
                precision_loss["empirical_snr_db"] = float(empirical_snr)
                
                # Calculate bit error rate equivalent
                precision_loss["relative_error"] = float(np.sqrt(noise_power) / (np.std(original_tensor) + 1e-10))
        
        return precision_loss

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quantization analysis."""
        recommendations = []
        
        # Check for saturation artifacts
        if results.get("artifacts", {}).get("saturation", {}).get("has_significant_saturation", False):
            sat_data = results["artifacts"]["saturation"]
            recommendations.append(
                f"Significant saturation detected ({sat_data['min_saturation_percent']:.1f}% at min, "
                f"{sat_data['max_saturation_percent']:.1f}% at max). Consider increasing quantization range."
            )
        
        # Check for high precision loss
        precision_loss = results.get("precision_loss", {})
        if "empirical_snr_db" in precision_loss and precision_loss["empirical_snr_db"] < 20:
            recommendations.append(
                f"Low SNR detected ({precision_loss['empirical_snr_db']:.1f} dB). "
                "Consider using higher bit depth or different quantization scheme."
            )
        
        # Check for clustering artifacts
        clustering = results.get("artifacts", {}).get("value_clustering", {})
        if clustering.get("has_significant_clustering", False) and clustering["clustering_strength"] > 10:
            recommendations.append(
                "Strong value clustering detected. This may indicate over-quantization. "
                "Consider using more quantization levels or asymmetric quantization."
            )
        
        # Check for banding artifacts
        banding = results.get("artifacts", {}).get("banding", {})
        if banding.get("has_banding", False):
            recommendations.append(
                f"Banding artifacts detected (score: {banding['banding_score']:.2f}). "
                "Consider using dithering or higher bit depth."
            )
        
        # Check for high frequency artifacts
        freq_analysis = results.get("frequency_analysis", {})
        if freq_analysis.get("has_high_freq_artifacts", False):
            recommendations.append(
                "High-frequency artifacts detected in spectrum. "
                "This may indicate quantization noise. Consider low-pass filtering or higher precision."
            )
        
        # Scheme-specific recommendations
        scheme = results.get("detected_scheme", "")
        if scheme == "symmetric_uniform" and results.get("zero_bias", {}).get("has_zero_bias", False):
            recommendations.append(
                "Zero bias detected with symmetric quantization. "
                "Consider switching to asymmetric quantization for better precision."
            )
        
        if not recommendations:
            recommendations.append("No significant quantization artifacts detected. Quality appears acceptable.")
        
        return recommendations

    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence in the quantization analysis."""
        confidence_factors = []
        
        # Scheme detection confidence
        if "scheme_detection_confidence" in results:
            confidence_factors.append(results["scheme_detection_confidence"])
        
        # Artifact detection confidence
        artifacts = results.get("artifacts", {})
        for artifact_type, artifact_data in artifacts.items():
            if "confidence" in artifact_data:
                confidence_factors.append(artifact_data["confidence"])
        
        # Tensor statistics confidence (based on sample size)
        tensor_stats = results.get("tensor_stats", {})
        if "unique_values" in tensor_stats:
            unique_values = tensor_stats["unique_values"]
            # More unique values generally means higher confidence in analysis
            stats_confidence = min(1.0, unique_values / 100.0)
            confidence_factors.append(stats_confidence)
        
        # Calculate weighted average confidence
        if confidence_factors:
            return float(np.mean(confidence_factors))
        else:
            return 0.5  # Default moderate confidence

    def _calculate_quality_metrics(self, tensor: np.ndarray, 
            original_tensor: np.ndarray = None,
            scheme_data: Dict[str, Any] = None) -> Dict[str, Any]:
     """Calculate quality metrics for the quantized tensor."""
     metrics = {}
     
     # If original tensor is available, calculate direct comparison metrics
     if original_tensor is not None and tensor.shape == original_tensor.shape:
      # Calculate Mean Squared Error
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
     
     # Calculate metrics based on tensor properties alone
     # Dynamic range
     tensor_range = tensor.max() - tensor.min()
     metrics["dynamic_range"] = float(tensor_range)
     
     # Bit utilization (how well the available quantization levels are used)
     unique_values = len(np.unique(tensor))
     if scheme_data and "estimated_bits" in scheme_data:
      max_possible_values = 2 ** scheme_data["estimated_bits"]
      metrics["bit_utilization"] = float(unique_values / max_possible_values)
     
     # Entropy-based quality assessment
     if unique_values > 1:
      # Calculate Shannon entropy
      _, counts = np.unique(tensor, return_counts=True)
      probabilities = counts / np.sum(counts)
      entropy = -np.sum(probabilities * np.log2(probabilities))
      metrics["entropy"] = float(entropy)
      
      # Theoretical maximum entropy for this bit depth
      if scheme_data and "estimated_bits" in scheme_data:
       max_entropy = scheme_data["estimated_bits"]
       metrics["entropy_efficiency"] = float(entropy / max_entropy)
     
     # Quantization noise estimate (based on step size)
     if tensor_range > 0 and unique_values > 1:
      avg_step = tensor_range / (unique_values - 1)
      # Assume uniform distribution of quantization error
      quantization_noise_var = (avg_step ** 2) / 12
      metrics["quantization_noise_variance"] = float(quantization_noise_var)
      metrics["quantization_noise_std"] = float(np.sqrt(quantization_noise_var))
     
     return metrics


class QuantizationSchemeDetector:
    """Advanced quantization scheme detection using statistical analysis."""
    
    def __init__(self):
        self.supported_schemes = [
            QuantizationType.SYMMETRIC_UNIFORM,
            QuantizationType.ASYMMETRIC_UNIFORM,
            QuantizationType.BLOCKWISE,
            QuantizationType.GROUPED,
            QuantizationType.K_QUANTS
        ]
        self.k_quants_types = [
            KQuantsType.Q2_K,
            KQuantsType.Q3_K,
            KQuantsType.Q4_K,
            KQuantsType.Q5_K,
            KQuantsType.Q6_K,
            KQuantsType.Q8_K
        ]
    
    def detect_scheme(self, tensor: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect quantization scheme from tensor data and optional metadata.
        
        Args:
            tensor: Input tensor to analyze
            metadata: Optional metadata dictionary containing hints about quantization
            
        Returns:
            Dictionary with detection results including scheme type, parameters, and confidence
        """
        if tensor.size == 0:
            return {
                "scheme": QuantizationType.NONE,
                "confidence": 0.0,
                "parameters": {},
                "error": "Empty tensor"
            }
        
        # Initialize results
        results = {
            "scheme": QuantizationType.NONE,
            "confidence": 0.0,
            "parameters": {},
            "analysis": {}
        }
        
        # Analyze tensor statistics
        tensor_stats = self._analyze_tensor_statistics(tensor)
        results["analysis"]["tensor_stats"] = tensor_stats
        
        # Check for obvious patterns first
        if self._is_likely_float(tensor):
            results["scheme"] = QuantizationType.NONE
            results["confidence"] = 0.9
            results["parameters"] = {"reason": "Float-like values detected"}
            return results
        
        # Test each quantization scheme
        scheme_scores = {}
        
        # Test symmetric uniform
        symmetric_score = self._test_symmetric_uniform(tensor, tensor_stats)
        scheme_scores[QuantizationType.SYMMETRIC_UNIFORM] = symmetric_score
        
        # Test asymmetric uniform
        asymmetric_score = self._test_asymmetric_uniform(tensor, tensor_stats)
        scheme_scores[QuantizationType.ASYMMETRIC_UNIFORM] = asymmetric_score
        
        # Test blockwise (requires 2D+ tensor)
        if tensor.ndim >= 2:
            blockwise_score = self._test_blockwise(tensor, tensor_stats)
            scheme_scores[QuantizationType.BLOCKWISE] = blockwise_score
        
        # Test grouped (requires sufficient channels)
        if tensor.ndim >= 2 and tensor.shape[0] >= 8:
            grouped_score = self._test_grouped(tensor, tensor_stats)
            scheme_scores[QuantizationType.GROUPED] = grouped_score
        
        # Test k-quants (requires specific metadata or patterns)
        if metadata and "quantization_type" in metadata:
            kquants_score = self._test_k_quants(tensor, tensor_stats, metadata)
            scheme_scores[QuantizationType.K_QUANTS] = kquants_score
        
        # Find best scheme
        best_scheme = max(scheme_scores, key=lambda x: scheme_scores[x]["confidence"])
        results["scheme"] = best_scheme
        results["confidence"] = scheme_scores[best_scheme]["confidence"]
        results["parameters"] = scheme_scores[best_scheme]["parameters"]
        results["analysis"]["scheme_scores"] = scheme_scores
        
        return results
    
    def _analyze_tensor_statistics(self, tensor: np.ndarray) -> Dict[str, Any]:
        """Analyze basic tensor statistics for scheme detection."""
        flat_tensor = tensor.flatten()
        unique_values = np.unique(flat_tensor)
        
        stats = {
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
            "unique_count": len(unique_values),
            "range": float(tensor.max() - tensor.min()),
            "is_integer": np.all(np.equal(np.mod(flat_tensor, 1), 0)),
            "has_negative": bool(tensor.min() < 0),
            "zero_centered": bool(abs(tensor.mean()) < 0.1 * tensor.std())
        }
        
        # Calculate value distribution metrics
        if len(unique_values) > 1:
            # Check spacing between values
            spacing = np.diff(np.sort(unique_values))
            stats["avg_spacing"] = float(np.mean(spacing))
            stats["spacing_std"] = float(np.std(spacing))
            stats["uniform_spacing"] = bool(stats["spacing_std"] < 0.1 * stats["avg_spacing"])
        
        return stats
    
    def _is_likely_float(self, tensor: np.ndarray) -> bool:
        """Check if tensor contains float-like values (not quantized)."""
        if tensor.dtype in [np.float16, np.float32, np.float64]:
            # Check for too many unique values for typical quantization
            unique_count = len(np.unique(tensor))
            if unique_count > 256:  # Too many unique values for typical quantization
                return True
            
            # Check for non-uniform spacing
            unique_values = np.unique(tensor)
            if len(unique_values) > 10:
                spacing = np.diff(unique_values)
                cv = np.std(spacing) / (np.mean(spacing) + 1e-10)
                if cv > 0.5:  # High coefficient of variation in spacing
                    return True
        
        return False
    
    def _test_symmetric_uniform(self, tensor: np.ndarray, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Test if tensor matches symmetric uniform quantization pattern."""
        result = {"confidence": 0.0, "parameters": {}}
        
        # Symmetric quantization should be roughly zero-centered
        if not stats["zero_centered"]:
            result["confidence"] = 0.1
            result["parameters"]["reason"] = "Not zero-centered"
            return result
        
        # Should have uniform spacing
        confidence = 0.5
        if stats.get("uniform_spacing", False):
            confidence += 0.3
        
        # Should have integer values for typical quantization
        if stats["is_integer"]:
            confidence += 0.2
        
        # Check for power-of-2 number of unique values
        unique_count = stats["unique_count"]
        if unique_count > 0 and (unique_count & (unique_count - 1)) == 0:
            confidence += 0.1
        
        # Estimate parameters
        if stats["range"] > 0:
            estimated_bits = max(1, int(np.ceil(np.log2(unique_count))))
            scale = stats["range"] / (2 ** estimated_bits - 1)
            result["parameters"] = {
                "estimated_bits": estimated_bits,
                "scale": scale,
                "zero_point": 0
            }
        
        result["confidence"] = min(confidence, 1.0)
        return result
    
    def _test_asymmetric_uniform(self, tensor: np.ndarray, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Test if tensor matches asymmetric uniform quantization pattern."""
        result = {"confidence": 0.0, "parameters": {}}
        
        # Asymmetric quantization can have any center
        confidence = 0.4
        
        # Should have uniform spacing
        if stats.get("uniform_spacing", False):
            confidence += 0.3
        
        # Should have integer values for typical quantization
        if stats["is_integer"]:
            confidence += 0.2
        
        # Check for power-of-2 number of unique values
        unique_count = stats["unique_count"]
        if unique_count > 0 and (unique_count & (unique_count - 1)) == 0:
            confidence += 0.1
        
        # Estimate parameters
        if stats["range"] > 0:
            estimated_bits = max(1, int(np.ceil(np.log2(unique_count))))
            scale = stats["range"] / (2 ** estimated_bits - 1)
            zero_point = int(-stats["min"] / scale) if scale > 0 else 0
            result["parameters"] = {
                "estimated_bits": estimated_bits,
                "scale": scale,
                "zero_point": zero_point
            }
        
        result["confidence"] = min(confidence, 1.0)
        return result
    
    def _test_blockwise(self, tensor: np.ndarray, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Test if tensor matches blockwise quantization pattern."""
        result = {"confidence": 0.0, "parameters": {}}
        
        # Blockwise quantization shows variation in statistics across blocks
        block_sizes = [8, 16, 32, 64]
        best_score = 0.0
        best_block_size = 32
        
        for block_size in block_sizes:
            score = self._analyze_blockwise_pattern(tensor, block_size)
            if score > best_score:
                best_score = score
                best_block_size = block_size
        
        result["confidence"] = best_score
        result["parameters"] = {
            "block_size": best_block_size,
            "block_dim": -1  # Flatten-then-block by default
        }
        
        return result
    
    def _test_grouped(self, tensor: np.ndarray, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Test if tensor matches grouped quantization pattern."""
        result = {"confidence": 0.0, "parameters": {}}
        
        # Grouped quantization shows variation in statistics across channel groups
        group_sizes = [4, 8, 16, 32, 64]
        best_score = 0.0
        best_group_size = 32
        
        for group_size in group_sizes:
            score = self._analyze_grouped_pattern(tensor, group_size)
            if score > best_score:
                best_score = score
                best_group_size = group_size
        
        result["confidence"] = best_score
        result["parameters"] = {
            "group_size": best_group_size,
            "channel_dim": 0  # Assume first dimension is channels
        }
        
        return result
    
    def _test_k_quants(self, tensor: np.ndarray, stats: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Test if tensor matches k-quants quantization pattern."""
        result = {"confidence": 0.0, "parameters": {}}
        
        # Check metadata for k-quants indicators
        quant_type = metadata.get("quantization_type", "").lower()
        if any(ktype.value in quant_type for ktype in self.k_quants_types):
            result["confidence"] = 0.8
            result["parameters"] = {"quant_type": quant_type}
        
        # Check tensor structure for k-quants patterns
        elif self._has_k_quants_structure(tensor):
            result["confidence"] = 0.6
            result["parameters"] = {"quant_type": "q4_k"}  # Default assumption
        
        return result
    
    def _analyze_blockwise_pattern(self, tensor: np.ndarray, block_size: int) -> float:
        """Analyze tensor for blockwise quantization patterns."""
        if tensor.size < block_size * 2:
            return 0.0
        
        # Flatten tensor and analyze blocks
        flat_tensor = tensor.flatten()
        block_stats = []
        
        for i in range(0, len(flat_tensor), block_size):
            block = flat_tensor[i:i+block_size]
            if len(block) >= block_size // 2:  # At least half block size
                block_stats.append({
                    "min": block.min(),
                    "max": block.max(),
                    "mean": block.mean(),
                    "std": block.std()
                })
        
        if len(block_stats) < 2:
            return 0.0
        
        # Calculate variation in block statistics
        block_stats = np.array([[s["min"], s["max"], s["mean"], s["std"]] for s in block_stats])
        
        # Calculate coefficient of variation for each statistic
        cv_scores = []
        for i in range(4):
            col = block_stats[:, i]
            if np.mean(col) != 0:
                cv = np.std(col) / np.abs(np.mean(col))
                cv_scores.append(cv)
        
        # Higher CV indicates more likely blockwise quantization
        avg_cv = np.mean(cv_scores) if cv_scores else 0.0
        return min(avg_cv, 1.0)
    
    def _analyze_grouped_pattern(self, tensor: np.ndarray, group_size: int) -> float:
        """Analyze tensor for grouped quantization patterns."""
        if tensor.shape[0] < group_size * 2:
            return 0.0
        
        # Analyze groups of channels
        group_stats = []
        for i in range(0, tensor.shape[0], group_size):
            group = tensor[i:i+group_size]
            if group.shape[0] >= group_size // 2:
                group_stats.append({
                    "min": group.min(),
                    "max": group.max(),
                    "mean": group.mean(),
                    "std": group.std()
                })
        
        if len(group_stats) < 2:
            return 0.0
        
        # Calculate variation in group statistics
        group_stats = np.array([[s["min"], s["max"], s["mean"], s["std"]] for s in group_stats])
        
        # Calculate coefficient of variation for each statistic
        cv_scores = []
        for i in range(4):
            col = group_stats[:, i]
            if np.mean(col) != 0:
                cv = np.std(col) / np.abs(np.mean(col))
                cv_scores.append(cv)
        
        # Higher CV indicates more likely grouped quantization
        avg_cv = np.mean(cv_scores) if cv_scores else 0.0
        return min(avg_cv, 1.0)
    
    def _has_k_quants_structure(self, tensor: np.ndarray) -> bool:
        """Check if tensor has structure typical of k-quants quantization."""
        # k-quants typically have specific block sizes and patterns
        # This is a simplified check - real k-quants detection would be more complex
        
        # Check if tensor size is compatible with k-quants block structures
        common_block_sizes = [24, 32, 48, 64]  # Common k-quants block sizes in bytes
        
        for block_size in common_block_sizes:
            if tensor.size % block_size == 0:
                return True
        
        return False


def create_quantization_reconstructor(quantization_info: Dict[str, Any]) -> QuantizationReconstructor:
    """
    Factory function to create a QuantizationReconstructor with validated configuration.
    
    Args:
        quantization_info: Dictionary containing quantization parameters
        
    Returns:
        Configured QuantizationReconstructor instance
        
    Raises:
        ValueError: If quantization_info is invalid
    """
    # Validate required fields
    if not isinstance(quantization_info, dict):
        raise ValueError("quantization_info must be a dictionary")
    
    # Set default values for missing fields
    defaults = {
        "scheme": "symmetric_uniform",
        "bits": 8,
        "scale": 1.0,
        "zero_point": 0,
        "layer_specific": {}
    }
    
    # Merge with defaults
    config = {**defaults, **quantization_info}
    
    # Validate scheme
    valid_schemes = ["symmetric_uniform", "asymmetric_uniform", "blockwise", "grouped", "k_quants", "custom"]
    if config["scheme"] not in valid_schemes:
        logger.warning(f"Unknown quantization scheme: {config['scheme']}, using default")
        config["scheme"] = "symmetric_uniform"
    
    # Validate bits
    if not isinstance(config["bits"], int) or config["bits"] < 1 or config["bits"] > 32:
        logger.warning(f"Invalid bits value: {config['bits']}, using default")
        config["bits"] = 8
    
    return QuantizationReconstructor(config)


def detect_quantization_scheme(tensor: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function to detect quantization scheme from tensor.
    
    Args:
        tensor: Input tensor to analyze
        metadata: Optional metadata dictionary
        
    Returns:
        Dictionary with detection results
    """
    detector = QuantizationSchemeDetector()
    return detector.detect_scheme(tensor, metadata)


def analyze_quantization_quality(tensor: np.ndarray, 
                                original_tensor: np.ndarray = None,
                                quantization_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze quantization quality.
    
    Args:
        tensor: Quantized tensor to analyze
        original_tensor: Original unquantized tensor for comparison
        quantization_info: Known quantization parameters
        
    Returns:
        Dictionary with quality analysis results
    """
    # Auto-detect quantization scheme if not provided
    if quantization_info is None:
        detection_result = detect_quantization_scheme(tensor)
        quantization_info = {
            "scheme": detection_result["scheme"].value if hasattr(detection_result["scheme"], 'value') else str(detection_result["scheme"]),
            "bits": detection_result["parameters"].get("estimated_bits", 8),
            "scale": detection_result["parameters"].get("scale", 1.0),
            "zero_point": detection_result["parameters"].get("zero_point", 0)
        }
    
    # Create reconstructor and analyze
    reconstructor = create_quantization_reconstructor(quantization_info)
    return reconstructor.identify_artifacts(tensor, original_tensor)
