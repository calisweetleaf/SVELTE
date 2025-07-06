# src/tensor_analysis/tensor_field.py
"""
Tensor Field Constructor for SVELTE Framework.
Builds multi-dimensional tensor spaces and preserves relationships.
"""
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Set, Iterator, Callable
from dataclasses import dataclass
from collections import defaultdict
import scipy.sparse as sp
from enum import Enum
import warnings
from concurrent.futures import ThreadPoolExecutor
import os
import json
import h5py

# Configure logging
logger = logging.getLogger(__name__)

class TensorQuantizationType(Enum):
 """Enumeration of supported tensor quantization types."""
 NONE = "none"
 INT8 = "int8"
 UINT8 = "uint8"
 FP16 = "fp16"
 FP32 = "fp32"
 MIXED = "mixed"
 CUSTOM = "custom"

@dataclass
class TensorMetadata:
 """Metadata for tensors in the field."""
 name: str
 shape: Tuple[int, ...]
 dtype: np.dtype
 quantization: TensorQuantizationType
 block_size: int = 32
 sparsity: float = 0.0
 connectivity: Set[str] = None
 source: str = None
 description: str = None

class TensorRelationship:
 """Represents relationships between tensors in the field."""
 
 def __init__(self, source: str, target: str, relationship_type: str, weight: float = 1.0):
  """
  Initialize a relationship between two tensors.
  
  Args:
   source: Name of the source tensor
   target: Name of the target tensor
   relationship_type: Type of relationship (e.g., "attention", "residual", "feedforward")
   weight: Strength of the relationship
  """
  self.source = source
  self.target = target
  self.relationship_type = relationship_type
  self.weight = weight
  self.properties = {}
 
 def add_property(self, key: str, value: Any) -> None:
  """Add a custom property to the relationship."""
  self.properties[key] = value
  
 def to_dict(self) -> Dict[str, Any]:
  """Convert relationship to dictionary representation."""
  return {
   "source": self.source,
   "target": self.target,
   "type": self.relationship_type,
   "weight": self.weight,
   "properties": self.properties
  }

class TensorIndex:
 """Efficient indexing structure for tensor fields."""
 
 def __init__(self):
  """Initialize the tensor index."""
  self.name_index = {}
  self.shape_index = defaultdict(list)
  self.dtype_index = defaultdict(list)
  self.dimension_index = defaultdict(list)
  self.quantization_index = defaultdict(list)
  self.connectivity_graph = defaultdict(set)
  
 def add_tensor(self, name: str, metadata: TensorMetadata) -> None:
  """
  Add a tensor to the index.
  
  Args:
   name: Name of the tensor
   metadata: Tensor metadata
  """
  self.name_index[name] = metadata
  self.shape_index[metadata.shape].append(name)
  self.dtype_index[metadata.dtype].append(name)
  self.dimension_index[len(metadata.shape)].append(name)
  self.quantization_index[metadata.quantization].append(name)
  
  # Add connectivity information
  if metadata.connectivity:
   for connected_tensor in metadata.connectivity:
    self.connectivity_graph[name].add(connected_tensor)
    self.connectivity_graph[connected_tensor].add(name)
 
 def find_by_name(self, name: str) -> Optional[TensorMetadata]:
  """Find tensor by name."""
  return self.name_index.get(name)
 
 def find_by_shape(self, shape: Tuple[int, ...]) -> List[str]:
  """Find tensors with a specific shape."""
  return self.shape_index.get(shape, [])
 
 def find_by_dimensions(self, dims: int) -> List[str]:
  """Find tensors with a specific number of dimensions."""
  return self.dimension_index.get(dims, [])
 
 def get_connected_tensors(self, name: str) -> Set[str]:
  """Get all tensors connected to the given tensor."""
  return self.connectivity_graph.get(name, set())

class TensorField:
 """
 Represents a multi-dimensional field of tensors with preserved relationships.
 
 A TensorField maintains the structure and relationships between tensors,
 supporting efficient operations and analyses across the tensor space.
 """
 
 def __init__(self):
  """Initialize an empty tensor field."""
  self.tensors = {}
  self.metadata = {}
  self.relationships = []
  self.index = TensorIndex()
  self.cache = {}
  self._modified = False
  
 def add_tensor(self, name: str, tensor: np.ndarray, metadata: Optional[TensorMetadata] = None) -> None:
  """
  Add a tensor to the field.
  
  Args:
   name: Unique name for the tensor
   tensor: Tensor data as numpy array
   metadata: Optional metadata for the tensor
  
  Raises:
   ValueError: If a tensor with the same name already exists
  """
  if name in self.tensors:
   raise ValueError(f"Tensor '{name}' already exists in the field")
  
  self.tensors[name] = tensor
  
  # Create metadata if not provided
  if metadata is None:
   metadata = TensorMetadata(
    name=name,
    shape=tensor.shape,
    dtype=tensor.dtype,
    quantization=TensorQuantizationType.NONE,
    sparsity=self._calculate_sparsity(tensor)
   )
  
  self.metadata[name] = metadata
  self.index.add_tensor(name, metadata)
  self._modified = True
 
 def add_relationship(self, relationship: TensorRelationship) -> None:
  """Add a relationship between tensors."""
  if relationship.source not in self.tensors:
   raise ValueError(f"Source tensor '{relationship.source}' does not exist")
  if relationship.target not in self.tensors:
   raise ValueError(f"Target tensor '{relationship.target}' does not exist")
  
  self.relationships.append(relationship)
  
  # Update connectivity in metadata
  source_meta = self.metadata[relationship.source]
  target_meta = self.metadata[relationship.target]
  
  if source_meta.connectivity is None:
   source_meta.connectivity = set()
  if target_meta.connectivity is None:
   target_meta.connectivity = set()
   
  source_meta.connectivity.add(relationship.target)
  target_meta.connectivity.add(relationship.source)
  self._modified = True
 
 def get_tensor(self, name: str) -> np.ndarray:
  """
  Get a tensor by name.
  
  Args:
   name: Name of the tensor
   
  Returns:
   The tensor as a numpy array
   
  Raises:
   KeyError: If tensor does not exist
  """
  if name not in self.tensors:
   raise KeyError(f"Tensor '{name}' not found in field")
  return self.tensors[name]
 
 def get_metadata(self, name: str) -> TensorMetadata:
  """Get metadata for a tensor."""
  if name not in self.metadata:
   raise KeyError(f"Metadata for tensor '{name}' not found")
  return self.metadata[name]
 
 def get_relationships(self, tensor_name: str) -> List[TensorRelationship]:
  """Get all relationships involving a specific tensor."""
  return [r for r in self.relationships if r.source == tensor_name or r.target == tensor_name]
 
 def optimize_storage(self) -> None:
  """Optimize storage by applying appropriate compression based on tensor properties."""
  for name, tensor in list(self.tensors.items()):
   metadata = self.metadata[name]
   
   # Apply different optimization strategies based on tensor properties
   if metadata.sparsity > 0.7:
    # Convert to sparse format for highly sparse tensors
    sparse_tensor = sp.csr_matrix(tensor)
    self.tensors[name] = sparse_tensor
    logger.info(f"Converted tensor '{name}' to sparse format, sparsity: {metadata.sparsity:.2f}")
   
   elif metadata.quantization == TensorQuantizationType.FP16 and tensor.dtype != np.float16:
    # Quantize to FP16 if specified
    self.tensors[name] = tensor.astype(np.float16)
    logger.info(f"Quantized tensor '{name}' to FP16")
   
   elif metadata.quantization == TensorQuantizationType.INT8 and tensor.dtype != np.int8:
    # Quantize to INT8 if specified (with appropriate scaling)
    scale = np.max(np.abs(tensor)) / 127.0
    quantized = np.round(tensor / scale).astype(np.int8)
    self.tensors[name] = quantized
    # Store scale factor in metadata
    metadata.properties = getattr(metadata, "properties", {})
    metadata.properties["scale_factor"] = scale
    logger.info(f"Quantized tensor '{name}' to INT8 with scale {scale}")
  
  self._modified = True
 
 def _calculate_sparsity(self, tensor: np.ndarray) -> float:
  """Calculate sparsity of a tensor (ratio of zeros to total elements)."""
  if hasattr(tensor, "toarray"):  # For sparse matrices
   return 1.0 - (tensor.nnz / np.prod(tensor.shape))
  else:
   return np.count_nonzero(tensor == 0) / tensor.size
 
 def save(self, filepath: str, compression: bool = True) -> None:
  """
  Save the tensor field to disk.
  
  Args:
   filepath: Path to save the tensor field
   compression: Whether to apply compression
  """
  # Use HDF5 format for efficient storage of multiple tensors
  with h5py.File(filepath, 'w') as f:
   # Store tensors
   tensors_group = f.create_group('tensors')
   for name, tensor in self.tensors.items():
    if hasattr(tensor, "toarray"):  # Handle sparse tensors
     tensor = tensor.toarray()
    
    if compression:
     tensors_group.create_dataset(name, data=tensor, compression='gzip', compression_opts=9)
    else:
     tensors_group.create_dataset(name, data=tensor)
   
   # Store metadata
   metadata_group = f.create_group('metadata')
   for name, meta in self.metadata.items():
    meta_dict = {
     'shape': meta.shape,
     'dtype': str(meta.dtype),
     'quantization': meta.quantization.value,
     'block_size': meta.block_size,
     'sparsity': meta.sparsity,
     'source': meta.source,
     'description': meta.description
    }
    if meta.connectivity:
     meta_dict['connectivity'] = list(meta.connectivity)
     
    metadata_group.create_dataset(name, data=json.dumps(meta_dict))
   
   # Store relationships
   relationships_data = [r.to_dict() for r in self.relationships]
   f.create_dataset('relationships', data=json.dumps(relationships_data))
  
  self._modified = False
  logger.info(f"Tensor field saved to {filepath}")
 
 def load(self, filepath: str) -> None:
  """
  Load a tensor field from disk.
  
  Args:
   filepath: Path to the saved tensor field
  """
  with h5py.File(filepath, 'r') as f:
   # Load tensors
   tensors_group = f['tensors']
   for name in tensors_group.keys():
    self.tensors[name] = np.array(tensors_group[name])
   
   # Load metadata
   metadata_group = f['metadata']
   for name in metadata_group.keys():
    meta_dict = json.loads(metadata_group[name][()])
    
    connectivity = set(meta_dict.get('connectivity', [])) if 'connectivity' in meta_dict else None
    
    self.metadata[name] = TensorMetadata(
     name=name,
     shape=tuple(meta_dict['shape']),
     dtype=np.dtype(meta_dict['dtype']),
     quantization=TensorQuantizationType(meta_dict['quantization']),
     block_size=meta_dict.get('block_size', 32),
     sparsity=meta_dict.get('sparsity', 0.0),
     connectivity=connectivity,
     source=meta_dict.get('source'),
     description=meta_dict.get('description')
    )
    
    # Add to index
    self.index.add_tensor(name, self.metadata[name])
   
   # Load relationships
   if 'relationships' in f:
    relationships_data = json.loads(f['relationships'][()])
    for r_data in relationships_data:
     relationship = TensorRelationship(
      source=r_data['source'],
      target=r_data['target'],
      relationship_type=r_data['type'],
      weight=r_data['weight']
     )
     for key, value in r_data.get('properties', {}).items():
      relationship.add_property(key, value)
     self.relationships.append(relationship)
  
  self._modified = False
  logger.info(f"Tensor field loaded from {filepath}")
 
 def analyze(self) -> Dict[str, Any]:
  """
  Perform analysis on the tensor field.
  
  Returns:
   Dictionary of analysis results
  """
  results = {
   "tensor_count": len(self.tensors),
   "relationship_count": len(self.relationships),
   "total_parameters": sum(np.prod(t.shape) for t in self.tensors.values() if hasattr(t, "shape")),
   "memory_usage": sum(t.nbytes for t in self.tensors.values() if hasattr(t, "nbytes")),
   "tensor_dimensions": {},
   "tensor_types": {},
   "quantization_types": {}
  }
  
  # Analyze dimensions
  for meta in self.metadata.values():
   dims = len(meta.shape)
   results["tensor_dimensions"][dims] = results["tensor_dimensions"].get(dims, 0) + 1
   
   # Analyze dtypes
   dtype_str = str(meta.dtype)
   results["tensor_types"][dtype_str] = results["tensor_types"].get(dtype_str, 0) + 1
   
   # Analyze quantization
   quant = meta.quantization.value
   results["quantization_types"][quant] = results["quantization_types"].get(quant, 0) + 1
  
  return results

class TensorFieldConstructor:
 """
 Constructs a multi-dimensional tensor field from raw tensors.
 
 This class builds a structured representation of tensors, preserving their
 relationships and optimizing for storage and retrieval efficiency.
 """
 
 def __init__(self, tensors: Dict[str, Any], metadata: Optional[Dict[str, Dict[str, Any]]] = None):
  """
  Initialize the tensor field constructor.
  
  Args:
   tensors: Dictionary of raw tensors
   metadata: Optional dictionary of tensor metadata
  """
  self.tensors = tensors
  self.raw_metadata = metadata or {}
  self.tensor_field = TensorField()
  self.relationship_detector = TensorRelationshipDetector()
  self.quantization_mapper = QuantizationMapper()
  self._execution_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
 
 def construct(self) -> TensorField:
  """
  Build a multi-dimensional tensor field from raw tensors.
  
  This method processes all tensors, infers relationships,
  optimizes storage, and builds an indexed field structure.
  
  Returns:
   Constructed tensor field
  """
  logger.info(f"Starting construction of tensor field with {len(self.tensors)} tensors")
  
  # Process tensors and add to field
  for name, tensor_data in self.tensors.items():
   try:
    # Convert to numpy array if not already
    if not isinstance(tensor_data, np.ndarray):
     if hasattr(tensor_data, "numpy"):
      # For PyTorch or TensorFlow tensors
      tensor_data = tensor_data.numpy()
     else:
      tensor_data = np.array(tensor_data)
    
    # Create metadata
    metadata = self._create_metadata(name, tensor_data)
    
    # Add to field
    self.tensor_field.add_tensor(name, tensor_data, metadata)
    logger.debug(f"Added tensor '{name}' with shape {tensor_data.shape}")
   
   except Exception as e:
    logger.error(f"Failed to add tensor '{name}': {e}")
    raise
  
  # Detect relationships between tensors
  self._detect_relationships()
  
  # Apply quantization based on tensor properties
  self._apply_quantization()
  
  # Optimize storage
  self.tensor_field.optimize_storage()
  
  # Build index
  self._index()
  
  logger.info(f"Tensor field construction completed with {len(self.tensor_field.tensors)} tensors")
  return self.tensor_field
 
 def _create_metadata(self, name: str, tensor: np.ndarray) -> TensorMetadata:
  """Create metadata for a tensor."""
  # Use provided metadata if available
  if name in self.raw_metadata:
   raw_meta = self.raw_metadata[name]
   
   # Convert quantization string to enum
   quant_type = raw_meta.get("quantization", "none")
   try:
    quantization = TensorQuantizationType(quant_type)
   except ValueError:
    logger.warning(f"Unknown quantization type '{quant_type}' for tensor '{name}', using NONE")
    quantization = TensorQuantizationType.NONE
   
   # Create metadata with provided values
   return TensorMetadata(
    name=name,
    shape=tensor.shape,
    dtype=tensor.dtype,
    quantization=quantization,
    block_size=raw_meta.get("block_size", 32),
    sparsity=raw_meta.get("sparsity", self._calculate_sparsity(tensor)),
    connectivity=set(raw_meta.get("connectivity", [])) if "connectivity" in raw_meta else None,
    source=raw_meta.get("source"),
    description=raw_meta.get("description")
   )
  
  # Auto-detect metadata otherwise
  return TensorMetadata(
   name=name,
   shape=tensor.shape,
   dtype=tensor.dtype,
   quantization=self._infer_quantization(tensor),
   sparsity=self._calculate_sparsity(tensor),
   connectivity=None
  )
 
 def _calculate_sparsity(self, tensor: np.ndarray) -> float:
  """Calculate sparsity of a tensor."""
  return np.count_nonzero(tensor == 0) / tensor.size
 
 def _infer_quantization(self, tensor: np.ndarray) -> TensorQuantizationType:
  """Infer the most appropriate quantization type for a tensor."""
  if tensor.dtype in (np.float16, np.float32, np.float64):
   # For floating point, determine precision needs
   if tensor.size > 1000000:  # Large tensor
    return TensorQuantizationType.FP16
   return TensorQuantizationType.FP32
  
  elif tensor.dtype in (np.int8, np.uint8):
   # Already quantized
   return TensorQuantizationType.INT8 if tensor.dtype == np.int8 else TensorQuantizationType.UINT8
   
  return TensorQuantizationType.NONE
 
 def _detect_relationships(self) -> None:
  """Detect relationships between tensors in the field."""
  relationships = self.relationship_detector.detect(self.tensor_field)
  
  for relationship in relationships:
   try:
    self.tensor_field.add_relationship(relationship)
    logger.debug(f"Added relationship: {relationship.source} -> {relationship.target} ({relationship.relationship_type})")
   except ValueError as e:
    logger.warning(f"Failed to add relationship: {e}")
 
 def _apply_quantization(self) -> None:
  """Apply appropriate quantization to tensors."""
  for name, metadata in self.tensor_field.metadata.items():
   if metadata.quantization != TensorQuantizationType.NONE:
    continue  # Skip already quantized tensors
    
   tensor = self.tensor_field.get_tensor(name)
   quant_type, quant_params = self.quantization_mapper.get_optimal_quantization(tensor, name)
   
   # Update metadata
   metadata.quantization = quant_type
   
   logger.debug(f"Selected {quant_type.value} quantization for tensor '{name}'")
 
 def index(self) -> None:
  """
  Build a searchable index for the tensor field.
  
  The index enables efficient querying by tensor properties,
  including shape, dimensionality, and relationships.
  """
  self._index()
  return self.tensor_field.index
 
 def _index(self) -> None:
  """Internal method to build the tensor field index."""
  # Index is built incrementally as tensors are added
  logger.debug("Tensor field index built with "
     f"{len(self.tensor_field.index.name_index)} tensors, "
     f"{len(self.tensor_field.index.connectivity_graph)} connectivity entries")

class TensorRelationshipDetector:
 """Detects relationships between tensors based on names and structures."""
 
 def __init__(self):
  """Initialize the relationship detector."""
  self.name_patterns = [
   # Attention patterns
   (r'(.*)_query', r'\1_key', "attention_qk"),
   (r'(.*)_query', r'\1_value', "attention_qv"),
   (r'(.*)_key', r'\1_value', "attention_kv"),
   
   # Layer connections
   (r'layer_(\d+)', r'layer_(\d+)', "sequential"),
   
   # Model component connections
   (r'encoder', r'decoder', "encoder_decoder"),
  ]
 
 def detect(self, tensor_field: TensorField) -> List[TensorRelationship]:
  """
  Detect relationships between tensors in the field.
  
  Args:
   tensor_field: The tensor field to analyze
   
  Returns:
   List of detected relationships
  """
  relationships = []
  
  # First pass: name-based detection
  name_relationships = self._detect_by_name(tensor_field)
  relationships.extend(name_relationships)
  
  # Second pass: structure-based detection
  structure_relationships = self._detect_by_structure(tensor_field)
  relationships.extend(structure_relationships)
  
  # Third pass: dimension compatibility
  dimension_relationships = self._detect_by_dimensions(tensor_field)
  relationships.extend(dimension_relationships)
  
  return relationships
 
 def _detect_by_name(self, tensor_field: TensorField) -> List[TensorRelationship]:
  """Detect relationships based on tensor naming patterns."""
  import re
  relationships = []
  tensor_names = list(tensor_field.tensors.keys())
  
  for source_name in tensor_names:
   for target_name in tensor_names:
    if source_name == target_name:
     continue
     
    # Check for sequential layer pattern
    layer_match_source = re.search(r'layer_(\d+)', source_name)
    layer_match_target = re.search(r'layer_(\d+)', target_name)
    
    if layer_match_source and layer_match_target:
     source_idx = int(layer_match_source.group(1))
     target_idx = int(layer_match_target.group(1))
     
     if source_idx + 1 == target_idx:
      # Sequential layer relationship
      relationships.append(TensorRelationship(
       source=source_name,
       target=target_name,
       relationship_type="sequential_layer",
       weight=1.0
      ))
    
    # Check for attention component relationships
    if ("query" in source_name and "key" in target_name) or \
       ("key" in source_name and "value" in target_name) or \
       ("query" in source_name and "value" in target_name):
     # Extract base name without query/key/value component
     source_base = re.sub(r'_(query|key|value).*', '', source_name)
     target_base = re.sub(r'_(query|key|value).*', '', target_name)
     
     if source_base == target_base:
      rel_type = "attention_component"
      if "query" in source_name and "key" in target_name:
       rel_type = "attention_qk"
      elif "key" in source_name and "value" in target_name:
       rel_type = "attention_kv"
      elif "query" in source_name and "value" in target_name:
       rel_type = "attention_qv"
       
      relationships.append(TensorRelationship(
       source=source_name,
       target=target_name,
       relationship_type=rel_type,
       weight=1.0
      ))
  
  return relationships
 
 def _detect_by_structure(self, tensor_field: TensorField) -> List[TensorRelationship]:
  """Detect relationships based on tensor structure compatibility."""
  relationships = []
  tensor_names = list(tensor_field.tensors.keys())
  
  # Group tensors by dimensionality
  by_dims = defaultdict(list)
  for name in tensor_names:
   tensor = tensor_field.get_tensor(name)
   by_dims[len(tensor.shape)].append(name)
  
  # Look for compatible matrix multiplication pairs
  for name1 in tensor_names:
   tensor1 = tensor_field.get_tensor(name1)
   if len(tensor1.shape) < 2:
    continue
    
   for name2 in tensor_names:
    if name1 == name2:
     continue
     
    tensor2 = tensor_field.get_tensor(name2)
    if len(tensor2.shape) < 2:
     continue
    
    # Check for matrix multiplication compatibility
    if tensor1.shape[-1] == tensor2.shape[0]:
     relationships.append(TensorRelationship(
      source=name1,
      target=name2,
      relationship_type="matmul_compatible",
      weight=0.7
     ))
  
  return relationships
 
 def _detect_by_dimensions(self, tensor_field: TensorField) -> List[TensorRelationship]:
  """Detect relationships based on tensor dimension compatibility."""
  relationships = []
  tensor_names = list(tensor_field.tensors.keys())
  
  # Look for tensors with matching dimensions 
  for name1 in tensor_names:
   tensor1 = tensor_field.get_tensor(name1)
   shape1 = tensor1.shape
   
   for name2 in tensor_names:
    if name1 == name2:
     continue
     
    tensor2 = tensor_field.get_tensor(name2)
    shape2 = tensor2.shape
    
    # Check for exact shape match
    if shape1 == shape2:
     relationships.append(TensorRelationship(
      source=name1,
      target=name2,
      relationship_type="shape_identical",
      weight=0.5
     ))
    
    # Check for broadcast compatibility
    elif len(shape1) == len(shape2) and all(d1 == d2 or d1 == 1 or d2 == 1 
              for d1, d2 in zip(shape1, shape2)):
     relationships.append(TensorRelationship(
      source=name1,
      target=name2,
      relationship_type="broadcast_compatible",
      weight=0.4
     ))
  
  return relationships

class QuantizationMapper:
 """Maps tensors to appropriate quantization schemes based on characteristics."""
 
 def __init__(self):
  """Initialize the quantization mapper."""
  self.quantization_registry = {
   "weight_matrix": self._quantize_weight_matrix,
   "embedding": self._quantize_embedding,
   "attention": self._quantize_attention_matrix,
   "default": self._default_quantization
  }
 
 def get_optimal_quantization(self, tensor: np.ndarray, name: str) -> Tuple[TensorQuantizationType, Dict[str, Any]]:
  """
  Determine optimal quantization for a tensor.
  
  Args:
   tensor: The tensor to analyze
   name: Name of the tensor (used for heuristics)
   
  Returns:
   Tuple of (quantization_type, parameters)
  """
  # Determine tensor role from name
  tensor_role = self._determine_tensor_role(name, tensor)
  
  # Apply role-specific quantization
  if tensor_role in self.quantization_registry:
   return self.quantization_registry[tensor_role](tensor)
  
  return self.quantization_registry["default"](tensor)
 
 def _determine_tensor_role(self, name: str, tensor: np.ndarray) -> str:
  """Determine the role of a tensor based on name and properties."""
  name_lower = name.lower()
  
  if "weight" in name_lower or "kernel" in name_lower:
   if len(tensor.shape) >= 2:
    return "weight_matrix"
  
  if "embed" in name_lower:
   return "embedding"
   
  if "attn" in name_lower or "attention" in name_lower or any(x in name_lower for x in ["query", "key", "value"]):
   return "attention"
   
  return "default"
 
 def _quantize_weight_matrix(self, tensor: np.ndarray) -> Tuple[TensorQuantizationType, Dict[str, Any]]:
  """Quantization strategy for weight matrices."""
  # For larger matrices, use more aggressive quantization
  if tensor.size > 1000000:
   return TensorQuantizationType.INT8, {"scale_axis": 1}
  else:
   return TensorQuantizationType.FP16, {}
 
 def _quantize_embedding(self, tensor: np.ndarray) -> Tuple[TensorQuantizationType, Dict[str, Any]]:
  """Quantization strategy for embedding matrices."""
  # Embeddings are sensitive to quantization, prefer higher precision
  return TensorQuantizationType.FP16, {}
 
 def _quantize_attention_matrix(self, tensor: np.ndarray) -> Tuple[TensorQuantizationType, Dict[str, Any]]:
  """Quantization strategy for attention matrices."""
  if tensor.size > 500000:
   return TensorQuantizationType.FP16, {}
  else:
   # Keep smaller attention matrices at full precision
   return TensorQuantizationType.FP32, {}
 
 def _default_quantization(self, tensor: np.ndarray) -> Tuple[TensorQuantizationType, Dict[str, Any]]:
  """Default quantization strategy."""
  if tensor.dtype == np.float32 and tensor.size > 100000:
   return TensorQuantizationType.FP16, {}
  else:
   return TensorQuantizationType.NONE, {}
