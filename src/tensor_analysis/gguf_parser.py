# src/tensor_analysis/gguf_parser.py
"""
GGUF and GGML model file parser for SVELTE Framework.
Extracts tensor data, quantization schemes, and metadata.
"""
import struct
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from src.utils.file_io import read_binary_file, FileIOException
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GGUFParserError(Exception):
    """Exception raised for errors during GGUF file parsing."""
    pass

# GGUF format constants
class GGUFConstants:
    # Magic identifiers
    GGUF_MAGIC = 0x46554747  # "GGUF" in hex
    GGUF_VERSION_1 = 1
    GGUF_VERSION_2 = 2
    GGUF_VERSION_3 = 3
    
    # Data types
    DT_F32 = 0
    DT_F16 = 1
    DT_Q4_0 = 2
    DT_Q4_1 = 3
    DT_Q5_0 = 6
    DT_Q5_1 = 7
    DT_Q8_0 = 8
    DT_Q8_1 = 9
    DT_Q2_K = 10
    DT_Q3_K = 11
    DT_Q4_K = 12
    DT_Q5_K = 13
    DT_Q6_K = 14
    DT_Q8_K = 15
    DT_IQ2_XXS = 16
    DT_IQ2_XS = 17
    DT_IQ3_XXS = 18
    DT_IQ1_S = 19
    DT_IQ4_NL = 20
    DT_IQ3_S = 21
    DT_IQ2_S = 22
    DT_IQ4_XS = 23
    DT_I8 = 24
    DT_I16 = 25
    DT_I32 = 26
    DT_I64 = 27
    DT_F64 = 28
    DT_IQ1_M = 29
    
    # Metadata value types
    MD_TYPE_U8 = 0
    MD_TYPE_I8 = 1
    MD_TYPE_U16 = 2
    MD_TYPE_I16 = 3
    MD_TYPE_U32 = 4
    MD_TYPE_I32 = 5
    MD_TYPE_U64 = 6
    MD_TYPE_I64 = 7
    MD_TYPE_F32 = 8
    MD_TYPE_F64 = 9
    MD_TYPE_BOOL = 10
    MD_TYPE_STRING = 11
    MD_TYPE_ARRAY = 12
    
    # Type name mapping
    TYPE_NAMES = {
        DT_F32: "F32",
        DT_F16: "F16",
        DT_Q4_0: "Q4_0",
        DT_Q4_1: "Q4_1",
        DT_Q5_0: "Q5_0",
        DT_Q5_1: "Q5_1",
        DT_Q8_0: "Q8_0",
        DT_Q8_1: "Q8_1",
        DT_Q2_K: "Q2_K",
        DT_Q3_K: "Q3_K",
        DT_Q4_K: "Q4_K",
        DT_Q5_K: "Q5_K",
        DT_Q6_K: "Q6_K",
        DT_Q8_K: "Q8_K",
        DT_IQ2_XXS: "IQ2_XXS",
        DT_IQ2_XS: "IQ2_XS",
        DT_IQ3_XXS: "IQ3_XXS",
        DT_IQ1_S: "IQ1_S",
        DT_IQ4_NL: "IQ4_NL",
        DT_IQ3_S: "IQ3_S",
        DT_IQ2_S: "IQ2_S",
        DT_IQ4_XS: "IQ4_XS",
        DT_I8: "I8",
        DT_I16: "I16",
        DT_I32: "I32",
        DT_I64: "I64",
        DT_F64: "F64",
        DT_IQ1_M: "IQ1_M"
    }
    
    # Quantization block sizes (in bytes)
    BLOCK_SIZES = {
        DT_F32: 4,
        DT_F16: 2,
        DT_Q4_0: 20,  # 16 weights + 2 scale + 2 pad
        DT_Q4_1: 22,  # 16 weights + 2 scale + 2 min + 2 pad
        DT_Q5_0: 22,  # 16 weights + 4 high_bits + 2 scale
        DT_Q5_1: 24,  # 16 weights + 4 high_bits + 2 scale + 2 min
        DT_Q8_0: 34,  # 32 weights + 2 scale
        DT_Q8_1: 36,  # 32 weights + 2 scale + 2 min
        DT_Q2_K: 84,  # K-quant block
        DT_Q3_K: 110, # K-quant block
        DT_Q4_K: 144, # K-quant block
        DT_Q5_K: 176, # K-quant block
        DT_Q6_K: 210, # K-quant block
        DT_Q8_K: 256, # K-quant block
        DT_I8: 1,
        DT_I16: 2,
        DT_I32: 4,
        DT_I64: 8,
        DT_F64: 8,
    }

class GGUFParser:
    """
    Production-ready parser for GGUF (GGML Universal Format) model files.
    
    This class provides robust, memory-efficient extraction of tensor data,
    quantization information, and metadata from GGUF model files.
    """
    
    def __init__(self, filepath: str, memory_map: bool = True):
        """
        Initialize the GGUF parser.
        
        Args:
            filepath (str): Path to the GGUF model file
            memory_map (bool): Use memory mapping for large files
        """
        self.filepath = Path(filepath)
        self.memory_map = memory_map
        self.header = {}
        self.tensors = {}
        self.metadata = {}
        self.quantization = {}
        self._offset = 0
        self._data = None
        self._file_size = 0
        self._tensor_data_start = 0
        
        # Validate file exists and is readable
        if not self.filepath.exists():
            raise FileIOException(f"File not found: {filepath}")
        if not self.filepath.is_file():
            raise FileIOException(f"Path is not a file: {filepath}")
        
        self._file_size = self.filepath.stat().st_size
        logger.info(f"Initialized GGUF parser for {filepath} ({self._file_size:,} bytes)")
    
    def parse(self) -> Dict[str, Any]:
        """
        Parse the GGUF model file with comprehensive error handling.
        
        Returns:
            Dict[str, Any]: Complete parsed model information
        """
        try:
            self._data = read_binary_file(str(self.filepath))
            logger.info("File loaded into memory")
            
            # Parse in order: header -> metadata -> tensor info -> quantization analysis
            self._parse_header()
            logger.info(f"Header parsed: version {self.header['version']}, "
                       f"{self.header['tensor_count']} tensors, "
                       f"{self.header['metadata_count']} metadata entries")
            
            self._parse_metadata()
            logger.info(f"Metadata parsed: {len(self.metadata)} entries")
            
            self._parse_tensors()
            logger.info(f"Tensor info parsed: {len(self.tensors)} tensors")
            
            self._parse_quantization()
            logger.info("Quantization analysis completed")
            
            return {
                "header": self.header,
                "metadata": self.metadata,
                "tensors": {name: tensor["info"] for name, tensor in self.tensors.items()},
                "quantization": self.quantization,
                "file_info": {
                    "size": self._file_size,
                    "tensor_data_start": self._tensor_data_start
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to parse GGUF file: {e}")
            raise GGUFParserError(f"Failed to parse GGUF file: {e}") from e
        finally:
            # Clear data to free memory
            self._data = None
            logger.info("Memory cleared")

    def _read_bytes(self, size: int) -> bytes:
        """Read bytes with bounds checking."""
        if self._offset + size > len(self._data):
            raise GGUFParserError(
                f"Attempt to read past end of file "
                f"(offset {self._offset}, size {size}, file size {len(self._data)})"
            )
        
        result = self._data[self._offset:self._offset + size]
        self._offset += size
        return result

    def _read_u8(self) -> int:
        """Read an unsigned 8-bit integer."""
        return struct.unpack("<B", self._read_bytes(1))[0]

    def _read_i8(self) -> int:
        """Read a signed 8-bit integer."""
        return struct.unpack("<b", self._read_bytes(1))[0]

    def _read_u16(self) -> int:
        """Read an unsigned 16-bit integer."""
        return struct.unpack("<H", self._read_bytes(2))[0]

    def _read_i16(self) -> int:
        """Read a signed 16-bit integer."""
        return struct.unpack("<h", self._read_bytes(2))[0]

    def _read_u32(self) -> int:
        """Read an unsigned 32-bit integer."""
        return struct.unpack("<I", self._read_bytes(4))[0]

    def _read_i32(self) -> int:
        """Read a signed 32-bit integer."""
        return struct.unpack("<i", self._read_bytes(4))[0]

    def _read_u64(self) -> int:
        """Read an unsigned 64-bit integer."""
        return struct.unpack("<Q", self._read_bytes(8))[0]

    def _read_i64(self) -> int:
        """Read a signed 64-bit integer."""
        return struct.unpack("<q", self._read_bytes(8))[0]

    def _read_f32(self) -> float:
        """Read a 32-bit float."""
        return struct.unpack("<f", self._read_bytes(4))[0]

    def _read_f64(self) -> float:
        """Read a 64-bit float."""
        return struct.unpack("<d", self._read_bytes(8))[0]

    def _read_bool(self) -> bool:
        """Read a boolean value."""
        return self._read_u8() != 0

    def _read_string(self) -> str:
        """Read a length-prefixed string with validation."""
        length = self._read_u64()
        
        # Sanity check on string length
        if length > 1024 * 1024:  # 1MB limit
            raise GGUFParserError(f"String length too large: {length}")
        
        if length == 0:
            return ""
        
        try:
            return self._read_bytes(length).decode('utf-8')
        except UnicodeDecodeError as e:
            raise GGUFParserError(f"Invalid UTF-8 string: {e}") from e

    def _parse_header(self):
        """Parse GGUF header with validation."""
        if len(self._data) < 16:
            raise GGUFParserError("File too small to contain GGUF header")
        
        magic = struct.unpack("<I", self._data[0:4])[0]
        if magic != GGUFConstants.GGUF_MAGIC:
            raise GGUFParserError(f"Invalid GGUF magic: 0x{magic:08x}")
        
        self._offset = 4
        version = self._read_u32()
        
        if version not in [GGUFConstants.GGUF_VERSION_1, 
                          GGUFConstants.GGUF_VERSION_2, 
                          GGUFConstants.GGUF_VERSION_3]:
            raise GGUFParserError(f"Unsupported GGUF version: {version}")
        
        tensor_count = self._read_u64()
        metadata_count = self._read_u64()
        
        # Sanity checks
        if tensor_count > 100000:
            raise GGUFParserError(f"Unreasonable tensor count: {tensor_count}")
        if metadata_count > 10000:
            raise GGUFParserError(f"Unreasonable metadata count: {metadata_count}")
        
        self.header = {
            "magic": magic,
            "version": version,
            "tensor_count": tensor_count,
            "metadata_count": metadata_count,
        }

    def _parse_metadata_value(self, type_id: int) -> Any:
        """Parse a metadata value based on its type."""
        if type_id == GGUFConstants.MD_TYPE_U8:
            return self._read_u8()
        elif type_id == GGUFConstants.MD_TYPE_I8:
            return self._read_i8()
        elif type_id == GGUFConstants.MD_TYPE_U16:
            return self._read_u16()
        elif type_id == GGUFConstants.MD_TYPE_I16:
            return self._read_i16()
        elif type_id == GGUFConstants.MD_TYPE_U32:
            return self._read_u32()
        elif type_id == GGUFConstants.MD_TYPE_I32:
            return self._read_i32()
        elif type_id == GGUFConstants.MD_TYPE_U64:
            return self._read_u64()
        elif type_id == GGUFConstants.MD_TYPE_I64:
            return self._read_i64()
        elif type_id == GGUFConstants.MD_TYPE_F32:
            return self._read_f32()
        elif type_id == GGUFConstants.MD_TYPE_F64:
            return self._read_f64()
        elif type_id == GGUFConstants.MD_TYPE_BOOL:
            return self._read_bool()
        elif type_id == GGUFConstants.MD_TYPE_STRING:
            return self._read_string()
        elif type_id == GGUFConstants.MD_TYPE_ARRAY:
            return self._parse_array()
        else:
            logger.warning(f"Unknown metadata type: {type_id}")
            return f"<unknown type {type_id}>"

    def _parse_array(self) -> List[Any]:
        """Parse an array metadata value."""
        array_type = self._read_u32()
        array_len = self._read_u64()
        
        # Sanity check
        if array_len > 100000:
            raise GGUFParserError(f"Array too large: {array_len}")
        
        result = []
        for _ in range(array_len):
            try:
                value = self._parse_metadata_value(array_type)
                result.append(value)
            except Exception as e:
                logger.warning(f"Failed to parse array element: {e}")
                break
        
        return result

    def _parse_metadata(self):
        """Parse metadata section with comprehensive error handling."""
        for i in range(self.header["metadata_count"]):
            try:
                key = self._read_string()
                type_id = self._read_u32()
                value = self._parse_metadata_value(type_id)
                
                self.metadata[key] = value
                
                # Extract quantization info
                if key == "general.quantization_version":
                    self.quantization["version"] = value
                elif key.startswith("general.quantization_"):
                    param = key.split("general.quantization_")[1]
                    self.quantization[param] = value
                    
            except Exception as e:
                logger.warning(f"Failed to parse metadata entry {i}: {e}")
                continue

    def _parse_tensors(self):
        """Parse tensor information with validation."""
        tensor_infos = []
        
        for i in range(self.header["tensor_count"]):
            try:
                name = self._read_string()
                n_dims = self._read_u32()
                
                # Sanity check on dimensions
                if n_dims > 8:
                    raise GGUFParserError(f"Too many dimensions: {n_dims}")
                
                dims = []
                for _ in range(n_dims):
                    dim = self._read_u64()
                    if dim == 0:
                        raise GGUFParserError(f"Zero dimension in tensor {name}")
                    dims.append(dim)
                
                tensor_type = self._read_u32()
                offset = self._read_u64()
                
                # Validate tensor type
                if tensor_type not in GGUFConstants.TYPE_NAMES:
                    logger.warning(f"Unknown tensor type {tensor_type} for {name}")
                
                # Calculate tensor size
                elements = 1
                for dim in dims:
                    elements *= dim
                
                # Estimate size in bytes
                if tensor_type in GGUFConstants.BLOCK_SIZES:
                    if tensor_type in [GGUFConstants.DT_F32, GGUFConstants.DT_F16, 
                                     GGUFConstants.DT_I8, GGUFConstants.DT_I16,
                                     GGUFConstants.DT_I32, GGUFConstants.DT_I64, 
                                     GGUFConstants.DT_F64]:
                        size_bytes = elements * GGUFConstants.BLOCK_SIZES[tensor_type]
                    else:
                        # For quantized types, calculate based on block size
                        block_size = GGUFConstants.BLOCK_SIZES[tensor_type]
                        size_bytes = (elements * block_size) // 32  # Most quants are 32 elements per block
                else:
                    size_bytes = elements * 4  # Default assumption
                
                tensor_info = {
                    "name": name,
                    "dimensions": dims,
                    "type": tensor_type,
                    "type_name": GGUFConstants.TYPE_NAMES.get(tensor_type, f"Unknown-{tensor_type}"),
                    "offset": offset,
                    "elements": elements,
                    "size_bytes": size_bytes,
                    "n_dims": n_dims
                }
                
                tensor_infos.append(tensor_info)
                self.tensors[name] = {"info": tensor_info, "data": None}
                
            except Exception as e:
                logger.warning(f"Failed to parse tensor {i}: {e}")
                continue
        
        # Store tensor data start position
        self._tensor_data_start = self._offset
        
        # Store tensor types for quantization analysis
        self.quantization["tensor_types"] = {
            info["name"]: info["type_name"] for info in tensor_infos
        }

    def _parse_quantization(self):
        """Enhanced quantization analysis."""
        # Populate quantization details based on metadata and tensor types
        if not self.quantization.get("version"):
            # Determine quantization version from tensor types
            q_types = set(self.quantization["tensor_types"].values())
            has_k_quants = any(qt.endswith("_K") for qt in q_types)
            
            if has_k_quants:
                self.quantization["version"] = 2
            else:
                self.quantization["version"] = 1
            
        # Analyze quantization schemes
        q_schemes = {}
        total_params = 0
        
        for name, type_name in self.quantization["tensor_types"].items():
            tensor_info = self.tensors[name]["info"]
            
            if type_name not in q_schemes:
                q_schemes[type_name] = {
                    "count": 0,
                    "elements": 0,
                    "tensors": [],
                    "size_bytes": 0
                }
            
            q_schemes[type_name]["count"] += 1
            q_schemes[type_name]["elements"] += tensor_info["elements"]
            q_schemes[type_name]["tensors"].append(name)
            q_schemes[type_name]["size_bytes"] += tensor_info["size_bytes"]
            total_params += tensor_info["elements"]
        
        self.quantization["schemes"] = q_schemes
        self.quantization["total_parameters"] = total_params
        
        # Calculate compression ratio
        if total_params > 0:
            fp32_size = total_params * 4
            actual_size = sum(scheme["size_bytes"] for scheme in q_schemes.values())
            self.quantization["compression_ratio"] = fp32_size / actual_size if actual_size > 0 else 1.0
        
        # Identify precision levels
        precision_levels = set()
        for type_name in q_schemes.keys():
            if "Q2" in type_name:
                precision_levels.add("2-bit")
            elif "Q3" in type_name:
                precision_levels.add("3-bit")
            elif "Q4" in type_name:
                precision_levels.add("4-bit")
            elif "Q5" in type_name:
                precision_levels.add("5-bit")
            elif "Q6" in type_name:
                precision_levels.add("6-bit")
            elif "Q8" in type_name:
                precision_levels.add("8-bit")
            elif "F16" in type_name:
                precision_levels.add("16-bit")
            elif "F32" in type_name:
                precision_levels.add("32-bit")
        
        self.quantization["precision_levels"] = sorted(precision_levels)

    def load_tensor(self, name: str, max_elements: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load tensor data with proper quantization handling.
        
        Args:
            name: Tensor name
            max_elements: Maximum elements to load (None for all)
            
        Returns:
            Tuple of (numpy array, tensor info)
        """
        if name not in self.tensors:
            raise GGUFParserError(f"Tensor '{name}' not found")
        
        tensor_info = self.tensors[name]["info"]
        
        try:
            with open(self.filepath, 'rb') as f:
                f.seek(tensor_info["offset"])
                
                elements = tensor_info["elements"]
                if max_elements and max_elements < elements:
                    elements = max_elements
                
                tensor_type = tensor_info["type"]
                
                # Handle different tensor types
                if tensor_type == GGUFConstants.DT_F32:
                    data = f.read(elements * 4)
                    array = np.frombuffer(data, dtype=np.float32)
                elif tensor_type == GGUFConstants.DT_F16:
                    data = f.read(elements * 2)
                    array = np.frombuffer(data, dtype=np.float16)
                elif tensor_type == GGUFConstants.DT_I8:
                    data = f.read(elements)
                    array = np.frombuffer(data, dtype=np.int8)
                elif tensor_type == GGUFConstants.DT_I16:
                    data = f.read(elements * 2)
                    array = np.frombuffer(data, dtype=np.int16)
                elif tensor_type == GGUFConstants.DT_I32:
                    data = f.read(elements * 4)
                    array = np.frombuffer(data, dtype=np.int32)
                elif tensor_type == GGUFConstants.DT_I64:
                    data = f.read(elements * 8)
                    array = np.frombuffer(data, dtype=np.int64)
                elif tensor_type == GGUFConstants.DT_F64:
                    data = f.read(elements * 8)
                    array = np.frombuffer(data, dtype=np.float64)
                else:
                    # For quantized types, read raw bytes
                    if tensor_type in GGUFConstants.BLOCK_SIZES:
                        block_size = GGUFConstants.BLOCK_SIZES[tensor_type]
                        num_blocks = (elements + 31) // 32  # Most quants are 32 elements per block
                        data = f.read(num_blocks * block_size)
                    else:
                        # Fallback: read as bytes
                        data = f.read(elements)
                    array = np.frombuffer(data, dtype=np.uint8)
                
                # Reshape if possible
                if tensor_type in [GGUFConstants.DT_F32, GGUFConstants.DT_F16,
                                 GGUFConstants.DT_I8, GGUFConstants.DT_I16,
                                 GGUFConstants.DT_I32, GGUFConstants.DT_I64,
                                 GGUFConstants.DT_F64]:
                    try:
                        array = array.reshape(tensor_info["dimensions"])
                    except ValueError:
                        logger.warning(f"Could not reshape tensor {name} to {tensor_info['dimensions']}")
                
                return array, tensor_info
                
        except Exception as e:
            raise GGUFParserError(f"Failed to load tensor '{name}': {e}") from e

    def get_tensor(self, name: str) -> Any:
        """
        Get tensor information if available.
        
        Args:
            name (str): Tensor name
            
        Returns:
            Any: Tensor information or None if not found
        """
        return self.tensors.get(name)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dict[str, Any]: Model metadata dictionary
        """
        return self.metadata

    def get_quantization(self) -> Dict[str, Any]:
        """
        Get quantization scheme information.
        
        Returns:
            Dict[str, Any]: Quantization data dictionary
        """
        return self.quantization
    
    def get_tensor_names(self) -> list:
        """
        Get a list of all tensor names in the model.
        
        Returns:
            list: List of tensor names
        """
        return list(self.tensors.keys())
    
    def get_model_architecture(self) -> str:
        """
        Get the model architecture based on metadata.
        
        Returns:
            str: Architecture name or 'unknown'
        """
        return self.metadata.get("general.architecture", "unknown")
    
    def summary(self) -> Dict[str, Any]:
        """Generate comprehensive model summary."""
        tensor_types = {}
        total_tensors = len(self.tensors)
        total_parameters = 0
        total_size = 0
        
        for name, tensor in self.tensors.items():
            info = tensor["info"]
            type_name = info["type_name"]
            
            if type_name not in tensor_types:
                tensor_types[type_name] = {
                    "count": 0,
                    "elements": 0,
                    "size_bytes": 0,
                    "tensors": []
                }
            
            tensor_types[type_name]["count"] += 1
            tensor_types[type_name]["elements"] += info["elements"]
            tensor_types[type_name]["size_bytes"] += info["size_bytes"]
            tensor_types[type_name]["tensors"].append(name)
            total_parameters += info["elements"]
            total_size += info["size_bytes"]
        
        # Calculate memory efficiency
        fp32_size = total_parameters * 4
        compression_ratio = fp32_size / total_size if total_size > 0 else 1.0
        
        return {
            "model_type": self.metadata.get("general.architecture", "Unknown"),
            "model_name": self.metadata.get("general.name", "Unknown"),
            "quantization_version": self.quantization.get("version", "Unknown"),
            "tensor_count": total_tensors,
            "parameter_count": total_parameters,
            "model_size_bytes": total_size,
            "compression_ratio": compression_ratio,
            "tensor_types": tensor_types,
            "precision_levels": self.quantization.get("precision_levels", []),
            "metadata_keys": list(self.metadata.keys()),
            "file_size": self._file_size,
            "architecture_details": {
                "vocab_size": self.metadata.get("llama.vocab_size", 0),
                "context_length": self.metadata.get("llama.context_length", 0),
                "embedding_length": self.metadata.get("llama.embedding_length", 0),
                "block_count": self.metadata.get("llama.block_count", 0),
                "feed_forward_length": self.metadata.get("llama.feed_forward_length", 0),
                "attention_head_count": self.metadata.get("llama.attention.head_count", 0),
                "attention_head_count_kv": self.metadata.get("llama.attention.head_count_kv", 0),
                "rope_theta": self.metadata.get("llama.rope.theta", 0),
            }
        }

def main():
    """Enhanced CLI with better error handling and options."""
    parser = argparse.ArgumentParser(description="SVELTE GGUF Parser CLI")
    parser.add_argument('model', type=str, help='Path to GGUF model file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--summary', '-s', action='store_true', help='Show model summary')
    parser.add_argument('--tensors', '-t', action='store_true', help='List all tensors')
    parser.add_argument('--metadata', '-m', action='store_true', help='Show metadata')
    parser.add_argument('--quantization', '-q', action='store_true', help='Show quantization info')
    parser.add_argument('--load-tensor', type=str, help='Load and display tensor data')
    parser.add_argument('--output', '-o', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize parser
        gguf_parser = GGUFParser(args.model)
        
        # Parse model
        result = gguf_parser.parse()
        
        # Display results based on flags
        if args.summary or not any([args.tensors, args.metadata, args.quantization]):
            print("=== Model Summary ===")
            summary = gguf_parser.summary()
            for key, value in summary.items():
                if key == "tensor_types":
                    print(f"{key}:")
                    for type_name, info in value.items():
                        print(f"  {type_name}: {info['count']} tensors, "
                              f"{info['elements']:,} parameters, "
                              f"{info['size_bytes']:,} bytes")
                else:
                    print(f"{key}: {value}")
        
        if args.tensors:
            print("\n=== Tensors ===")
            for name, info in result["tensors"].items():
                print(f"- {name}: {info['dimensions']} ({info['type_name']})")
        
        if args.metadata:
            print("\n=== Metadata ===")
            for key, value in result["metadata"].items():
                print(f"{key}: {value}")
        
        if args.quantization:
            print("\n=== Quantization ===")
            for key, value in result["quantization"].items():
                print(f"{key}: {value}")
        
        if args.load_tensor:
            print(f"\n=== Loading Tensor: {args.load_tensor} ===")
            try:
                tensor_data, tensor_info = gguf_parser.load_tensor(args.load_tensor, max_elements=100)
                print(f"Shape: {tensor_data.shape}")
                print(f"Dtype: {tensor_data.dtype}")
                print(f"First few values: {tensor_data.flat[:min(10, tensor_data.size)]}")
            except Exception as e:
                print(f"Error loading tensor: {e}")
        
        # Save results if requested
        if args.output:
            from src.utils.file_io import write_json
            write_json(result, args.output)
            print(f"\nResults saved to: {args.output}")
            
    except Exception as e:
        logger.error(f"Failed to process model: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
