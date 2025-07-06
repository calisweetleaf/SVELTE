# src/tensor_analysis/gguf_parser.py
"""
GGUF and GGML model file parser for SVELTE Framework.
Extracts tensor data, quantization schemes, and metadata.
"""
import struct
from typing import Dict, Any, Tuple
from src.utils.file_io import read_binary_file, FileIOException
import argparse

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
 
 # Metadata value types
 MD_TYPE_U8 = 0
 MD_TYPE_I8 = 1
 MD_TYPE_U16 = 2
 MD_TYPE_I16 = 3
 MD_TYPE_U32 = 4
 MD_TYPE_I32 = 5
 MD_TYPE_F32 = 6
 MD_TYPE_STRING = 7
 MD_TYPE_ARRAY = 8
 
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
  DT_Q8_K: "Q8_K"
 }

class GGUFParser:
 """
 Parser for GGUF (GGML Universal Format) model files.
 
 This class extracts tensor data, quantization information, and metadata
 from GGUF model files used by LLM frameworks like llama.cpp.
 
 Attributes:
  filepath (str): Path to the GGUF file
  header (dict): Parsed header information
  tensors (dict): Dictionary of extracted tensors
  metadata (dict): Model metadata
  quantization (dict): Quantization scheme information
 """
 
 def __init__(self, filepath: str):
  """
  Initialize the GGUF parser with a file path.
  
  Args:
   filepath (str): Path to the GGUF model file
  """
  self.filepath = filepath
  self.header = {}
  self.tensors = {}
  self.metadata = {}
  self.quantization = {}
  self._offset = 0
  self._data = None
  
 def parse(self) -> Dict[str, Any]:
  """
  Parse the GGUF model file and extract all information.
  
  Returns:
   Dict[str, Any]: Dictionary containing parsed model information
   
  Raises:
   FileIOException: If file cannot be read
   GGUFParserError: If parsing errors occur
  """
  try:
   self._data = read_binary_file(self.filepath)
   self._parse_header(self._data)
   self._parse_metadata(self._data)
   self._parse_tensors(self._data)
   self._parse_quantization(self._data)
   
   return {
    "header": self.header,
    "metadata": self.metadata,
    "tensors": {name: tensor["info"] for name, tensor in self.tensors.items()},
    "quantization": self.quantization
   }
  except FileIOException:
   raise
  except Exception as e:
   raise GGUFParserError(f"Failed to parse GGUF file: {e}") from e
  finally:
   # Clear the raw data to free memory
   self._data = None

 def _read_bytes(self, size: int) -> bytes:
  """Read bytes from the data buffer and advance offset."""
  if self._offset + size > len(self._data):
   raise GGUFParserError(f"Attempt to read past end of file (offset {self._offset}, size {size})")
  
  result = self._data[self._offset:self._offset + size]
  self._offset += size
  return result

 def _read_u32(self) -> int:
  """Read an unsigned 32-bit integer."""
  return struct.unpack("<I", self._read_bytes(4))[0]
 
 def _read_u64(self) -> int:
  """Read an unsigned 64-bit integer."""
  return struct.unpack("<Q", self._read_bytes(8))[0]
 
 def _read_f32(self) -> float:
  """Read a 32-bit float."""
  return struct.unpack("<f", self._read_bytes(4))[0]
 
 def _read_string(self) -> str:
  """Read a length-prefixed string."""
  length = self._read_u64()
  return self._read_bytes(length).decode('utf-8')
 
 def _parse_header(self, data: bytes):
  """
  Parse the GGUF header section.
  
  Args:
   data (bytes): Raw file data
   
  Raises:
   GGUFParserError: If header contains invalid data
  """
  magic = struct.unpack("<I", data[0:4])[0]
  if magic != GGUFConstants.GGUF_MAGIC:
   raise GGUFParserError(f"Invalid GGUF magic: {magic:#x}")
  
  self._offset = 4
  version = self._read_u32()
  
  if version not in [GGUFConstants.GGUF_VERSION_1, GGUFConstants.GGUF_VERSION_2, GGUFConstants.GGUF_VERSION_3]:
   raise GGUFParserError(f"Unsupported GGUF version: {version}")
  
  self.header = {
   "magic": magic,
   "version": version,
  }
  
  # Version-specific header fields
  tensor_count = self._read_u64()
  metadata_count = self._read_u64()
  
  self.header["tensor_count"] = tensor_count
  self.header["metadata_count"] = metadata_count
 
 def _parse_metadata(self, data: bytes):
  """
  Parse the metadata section of the GGUF file.
  
  Args:
   data (bytes): Raw file data
   
  Raises:
   GGUFParserError: If metadata parsing fails
  """
  for i in range(self.header["metadata_count"]):
   key = self._read_string()
   type_id = self._read_u32()
   
   if type_id == GGUFConstants.MD_TYPE_STRING:
    value = self._read_string()
   elif type_id == GGUFConstants.MD_TYPE_U32:
    value = self._read_u32()
   elif type_id == GGUFConstants.MD_TYPE_I32:
    value = struct.unpack("<i", self._read_bytes(4))[0]
   elif type_id == GGUFConstants.MD_TYPE_F32:
    value = self._read_f32()
   elif type_id == GGUFConstants.MD_TYPE_ARRAY:
    array_type = self._read_u32()
    array_len = self._read_u64()
    value = []
    
    # Parse array elements based on type
    for j in range(array_len):
     if array_type == GGUFConstants.MD_TYPE_F32:
      value.append(self._read_f32())
     elif array_type == GGUFConstants.MD_TYPE_U32:
      value.append(self._read_u32())
     elif array_type == GGUFConstants.MD_TYPE_STRING:
      value.append(self._read_string())
     else:
      # Skip unknown types
      self._offset += 4  # Assume most types are 4 bytes
   else:
    # For unknown types, store the type ID but no value
    value = f"<unknown type {type_id}>"
   
   self.metadata[key] = value
   
   # Detect quantization info in metadata
   if key == "general.quantization_version":
    self.quantization["version"] = value
   elif key.startswith("general.quantization_"):
    param = key.split("general.quantization_")[1]
    self.quantization[param] = value

 def _parse_tensors(self, data: bytes):
  """
  Parse tensor metadata and data from the GGUF file.
  
  Args:
   data (bytes): Raw file data
   
  Raises:
   GGUFParserError: If tensor parsing fails
  """
  tensor_infos = []
  
  # First pass: parse tensor information
  for i in range(self.header["tensor_count"]):
   name = self._read_string()
   
   n_dims = self._read_u32()
   dims = []
   for j in range(n_dims):
    dims.append(self._read_u64())
   
   tensor_type = self._read_u32()
   offset = self._read_u64()
   
   tensor_info = {
    "name": name,
    "dimensions": dims,
    "type": tensor_type,
    "type_name": GGUFConstants.TYPE_NAMES.get(tensor_type, f"Unknown-{tensor_type}"),
    "offset": offset,
    "elements": 1,
   }
   
   # Calculate number of elements
   for dim in dims:
    tensor_info["elements"] *= dim
   
   tensor_infos.append(tensor_info)
   self.tensors[name] = {"info": tensor_info, "data": None}
  
  # Store tensor offsets in the quantization info
  self.quantization["tensor_types"] = {info["name"]: info["type_name"] for info in tensor_infos}
  
 def _parse_quantization(self, data: bytes):
  """
  Extract and analyze quantization schemes from tensor data.
  
  Args:
   data (bytes): Raw file data
  """
  # Populate quantization details based on metadata and tensor types
  if not self.quantization.get("version"):
   # Determine quantization version from tensor types
   q_types = set(self.quantization["tensor_types"].values())
   has_k_quants = any(qt.endswith("_K") for qt in q_types)
   
   if has_k_quants:
    self.quantization["version"] = 2
   else:
    self.quantization["version"] = 1
    
  # Analyze quantization schemes used
  q_schemes = {}
  for name, type_name in self.quantization["tensor_types"].items():
   if type_name not in q_schemes:
    q_schemes[type_name] = []
   q_schemes[type_name].append(name)
   
  self.quantization["schemes"] = {
   scheme: {"count": len(tensors), "example": tensors[0]}
   for scheme, tensors in q_schemes.items()
  }
   
  # Extract block sizes for K-quants if available
  if "block_size" in self.metadata:
   self.quantization["block_size"] = self.metadata["block_size"]

 def load_tensor(self, name: str, max_elements: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
  """
  Load tensor data from the file.
  
  Args:
   name (str): Tensor name
   max_elements (int, optional): Maximum elements to load, for partial loading
   
  Returns:
   Tuple[np.ndarray, Dict[str, Any]]: Tensor data array and tensor info dictionary
   
  Raises:
   GGUFParserError: If tensor not found or loading fails
   FileIOException: If file reading fails
  """
  if name not in self.tensors:
   raise GGUFParserError(f"Tensor '{name}' not found in model")
  
  tensor_info = self.tensors[name]["info"]
  
  try:
   with open(self.filepath, 'rb') as f:
    f.seek(tensor_info["offset"])
    
    # Calculate elements to read
    elements = tensor_info["elements"]
    if max_elements and max_elements < elements:
     elements = max_elements
    
    # Determine bytes per element based on type
    if tensor_info["type"] == GGUFConstants.DT_F32:
     bytes_per_element = 4
     dtype = np.float32
    elif tensor_info["type"] == GGUFConstants.DT_F16:
     bytes_per_element = 2
     dtype = np.float16
    else:
     # For quantized types, we need to handle specially
     # In this example just loading raw bytes
     bytes_per_element = 1
     dtype = np.uint8
    
    # Read the data
    data = f.read(elements * bytes_per_element)
    
    # Convert to numpy array
    if tensor_info["type"] in [GGUFConstants.DT_F32, GGUFConstants.DT_F16]:
     array = np.frombuffer(data, dtype=dtype)
     # Reshape to original dimensions if possible
     if len(tensor_info["dimensions"]) > 0:
      array = array.reshape(tensor_info["dimensions"])
    else:
     # For now, just return raw data for quantized tensors
     array = np.frombuffer(data, dtype=np.uint8)
    
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
  """
  Generate a summary of the model.
  
  Returns:
   Dict[str, Any]: Model summary
  """
  tensor_types = {}
  total_tensors = len(self.tensors)
  total_parameters = 0
  
  for name, tensor in self.tensors.items():
   info = tensor["info"]
   type_name = info["type_name"]
   
   if type_name not in tensor_types:
    tensor_types[type_name] = {
     "count": 0,
     "elements": 0
    }
   
   tensor_types[type_name]["count"] += 1
   tensor_types[type_name]["elements"] += info["elements"]
   total_parameters += info["elements"]
   
  return {
   "model_type": self.metadata.get("general.architecture", "Unknown"),
   "quantization_version": self.quantization.get("version", "Unknown"),
   "tensor_count": total_tensors,
   "parameter_count": total_parameters,
   "tensor_types": tensor_types,
   "metadata_keys": list(self.metadata.keys())
  }

def main():
    parser = argparse.ArgumentParser(description="SVELTE GGUF Parser CLI")
    parser.add_argument('model', type=str, help='Path to GGUF model file')
    args = parser.parse_args()
    parser = GGUFParser(args.model)
    parser.parse()
    print("Tensors:")
    for name in parser.tensors:
        print(f"- {name}")
    print("Metadata:")
    print(parser.get_metadata())
    print("Quantization:")
    print(parser.get_quantization())

if __name__ == "__main__":
    main()
