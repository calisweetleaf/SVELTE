# src/metadata/metadata_extractor.py
"""
Metadata and Model Data Extractor for SVELTE Framework.
Extracts and validates model metadata, architecture, and provenance.
author: Morpheus
date: 2025-05-01
version: 1.0.0
description: This module is responsible for extracting metadata and binary data from a gguf model files.
It handles the extraction of model architecture, vocabulary, and provenance information.
It also validates the extracted data and prepares it for further analysis.
ID: 001
SHA-256: 3f3b2c4e5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a
"""
import hashlib
import logging
import os
import re
import json
import datetime
import mmap
import struct
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple, Union, BinaryIO, Iterator, Set
from pathlib import Path
from contextlib import contextmanager
import argparse
import sys

# Configure logging
logger = logging.getLogger("svelte.metadata")

class MetadataValidationError(Exception):
 """Raised when metadata validation fails."""
 
 def __init__(self, message: str, field: Optional[str] = None):
  """
  Initialize MetadataValidationError.
  
  Args:
   message: Error message describing the validation failure
   field: Optional field name that caused the validation error
  """
  super().__init__(message)
  self.field = field
  self.message = message

class GGUFFormatError(Exception):
 """Raised when GGUF file format is invalid or unsupported."""
 
 def __init__(self, message: str, version: Optional[int] = None):
  """
  Initialize GGUFFormatError.
  
  Args:
   message: Error message describing the format issue
   version: Optional version number that caused the error
  """
  super().__init__(message)
  self.version = version
  self.message = message

class DataExtractionError(Exception):
 """Raised when data extraction from GGUF file fails."""
 
 def __init__(self, message: str, offset: Optional[int] = None):
  """
  Initialize DataExtractionError.
  
  Args:
   message: Error message describing the extraction failure
   offset: Optional file offset where the error occurred
  """
  super().__init__(message)
  self.offset = offset
  self.message = message

class GGUFValueType(Enum):
 """GGUF value types based on GGUF specification."""
 UINT8 = 0
 INT8 = 1
 UINT16 = 2
 INT16 = 3
 UINT32 = 4
 INT32 = 5
 FLOAT32 = 6
 BOOL = 7
 STRING = 8
 ARRAY = 9
 UINT64 = 10
 INT64 = 11
 FLOAT64 = 12

class MetadataExtractor:
 """
 Extracts and validates metadata from GGUF model files.
 
 This class handles reading GGUF files, extracting metadata and binary data,
 validating the extracted information, and providing a clean interface to
 access the validated metadata.
 """
 
 # Magic number for GGUF format identification
 GGUF_MAGIC = b'GGUF'
 
 # Format version supported by this extractor
 SUPPORTED_VERSIONS = {1, 2, 3}
 
 # Required fields in metadata with their expected types
 REQUIRED_FIELDS = {
  "model_name": str,
  "architecture": dict,
  "vocabulary": list,
  "provenance": dict,
  "sha256": str,
  "created_at": str,
  "version": str,
 }
 
 # Architecture required fields
 ARCH_REQUIRED_FIELDS = {
  "layers": int,
  "hidden_size": int,
  "attention_heads": int,
  "vocab_size": int,
  "ffn_dim": int,
  "context_length": int,
  "activation_function": str
 }
 
 # Provenance required fields
 PROV_REQUIRED_FIELDS = {
  "source": str,
  "license": str,
  "authors": list,
  "creation_date": str,
  "description": str,
  "training_data": list,
  "parameters": dict
 }

 def __init__(self, source: Union[str, Path, Dict[str, Any]]):
  """
  Initialize the MetadataExtractor with a GGUF file path or raw metadata.
  
  Args:
   source: Either a path to a GGUF file or a dictionary containing raw metadata.
  """
  self.file_path = None
  self.file_size = 0
  self.raw_metadata = {}
  self.metadata: Dict[str, Any] = {}
  self.errors: List[str] = []
  self.binary_data: Dict[str, Any] = {}
  self.file_hash = ""
  
  if isinstance(source, (str, Path)):
   self.file_path = Path(source)
   if not self.file_path.exists():
    raise FileNotFoundError(f"GGUF file not found: {self.file_path}")
   if not self.file_path.is_file():
    raise ValueError(f"Not a file: {self.file_path}")
   self.file_size = self.file_path.stat().st_size
   logger.info(f"Initializing metadata extraction from file: {self.file_path}")
  elif isinstance(source, dict):
   self.raw_metadata = source
   logger.info("Initializing metadata extraction from provided dictionary")
  else:
   raise TypeError("Source must be either a file path or a metadata dictionary")

 def extract(self) -> Dict[str, Any]:
  """
  Extract and validate all metadata from the source.
  
  This method orchestrates the extraction process, handling both file-based
  and dictionary-based sources.
  
  Returns:
   Dict[str, Any]: The validated metadata dictionary.
   
  Raises:
   MetadataValidationError: If metadata validation fails.
   GGUFFormatError: If GGUF file format is invalid.
   DataExtractionError: If data extraction fails.
  """
  logger.info("Starting metadata extraction")
  
  try:
   if self.file_path:
    self._extract_from_file()
   else:
    self._extract_from_dict()
    
   self._validate_metadata()
   logger.info("Metadata extraction completed successfully")
   return self.metadata
   
  except (MetadataValidationError, GGUFFormatError, DataExtractionError) as e:
   logger.error(f"Extraction failed: {str(e)}")
   raise
  except Exception as e:
   error_msg = f"Unexpected error during metadata extraction: {str(e)}"
   logger.exception(error_msg)
   raise DataExtractionError(error_msg) from e

 def _extract_from_file(self) -> None:
  """
  Extract metadata from a GGUF file.
  
  This method handles reading the GGUF file format, extracting both
  metadata and binary tensor data.
  
  Raises:
   GGUFFormatError: If the file format is invalid.
   DataExtractionError: If data extraction fails.
  """
  if self.file_path is None:
   raise DataExtractionError("File path is not set")
   
  logger.info(f"Extracting metadata from file: {self.file_path}")
  
  # Calculate file hash in a memory-efficient way
  self.file_hash = self._calculate_file_hash()
  logger.debug(f"File SHA-256: {self.file_hash}")
  
  with open(self.file_path, 'rb') as f:
   # Use memory mapping for efficient handling of large files
   with self._memory_map_file(f) as mm:
    # Validate file format
    self._validate_gguf_format(mm)
    
    # Extract metadata
    metadata_offset, metadata_size = self._locate_metadata(mm)
    self.raw_metadata = self._parse_metadata(mm, metadata_offset, metadata_size)
    
    # Extract model architecture information
    architecture_offset = metadata_offset + metadata_size
    self.binary_data = self._extract_binary_data(mm, architecture_offset)
  
  # Merge binary architecture data into metadata
  if "architecture" not in self.raw_metadata:
   self.raw_metadata["architecture"] = {}
   
  for key, value in self.binary_data.items():
   if key.startswith("arch_"):
    clean_key = key[5:]  # Remove 'arch_' prefix
    self.raw_metadata["architecture"][clean_key] = value
  
  # Add file-related metadata
  self.raw_metadata["sha256"] = self.file_hash
  self.raw_metadata["file_size"] = self.file_size
  self.raw_metadata["file_path"] = str(self.file_path)
  
  # Continue with standard extraction process
  self._extract_from_dict()

 def _extract_from_dict(self) -> None:
  """
  Extract and validate metadata from a dictionary.
  
  This method processes the raw metadata dictionary, validating
  required fields and extracting structured metadata.
  
  Raises:
   MetadataValidationError: If metadata validation fails.
  """
  logger.info("Processing metadata from dictionary")
  self._validate_required_fields()
  
  # Extract and validate individual components
  self.metadata["model_name"] = self._extract_model_name()
  self.metadata["architecture"] = self._extract_architecture()
  self.metadata["vocabulary"] = self._extract_vocabulary()
  self.metadata["provenance"] = self._extract_provenance()
  self.metadata["sha256"] = self._extract_sha256()
  self.metadata["created_at"] = self._extract_created_at()
  self.metadata["version"] = self._extract_version()
  
  # Add any additional metadata fields that might be present
  for key, value in self.raw_metadata.items():
   if key not in self.metadata and key not in ['architecture', 'vocabulary', 'provenance']:
    self.metadata[key] = value

 @contextmanager
 def _memory_map_file(self, file_obj: BinaryIO) -> Iterator[mmap.mmap]:
  """
  Create a memory map for the file for efficient access.
  
  Args:
   file_obj: The binary file object to map.
   
  Yields:
   A memory-mapped file object.
  """
  mm = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
  try:
   yield mm
  finally:
   mm.close()

 def _calculate_file_hash(self) -> str:
  """
  Calculate SHA-256 hash of the file in a memory-efficient way.
  
  Returns:
   str: The hex digest of the file's SHA-256 hash.
  """
  if self.file_path is None:
   raise DataExtractionError("File path is not set")
   
  hash_sha256 = hashlib.sha256()
  with open(self.file_path, "rb") as f:
   # Read file in chunks to handle large files efficiently
   for chunk in iter(lambda: f.read(4096), b""):
    hash_sha256.update(chunk)
  return hash_sha256.hexdigest()

 def _validate_gguf_format(self, mm: mmap.mmap) -> None:
  """
  Validate that the file is a valid GGUF file.
  
  Args:
   mm: Memory-mapped file.
   
  Raises:
   GGUFFormatError: If the file is not a valid GGUF file.
  """
  # Check magic number
  magic = mm[:4]
  if magic != self.GGUF_MAGIC:
   raise GGUFFormatError(f"Invalid GGUF magic number: {magic}")
   
  # Check version
  version = struct.unpack('<I', mm[4:8])[0]
  if version not in self.SUPPORTED_VERSIONS:
   raise GGUFFormatError(f"Unsupported GGUF version: {version}")
   
  logger.debug(f"Valid GGUF format detected: version {version}")

 def _locate_metadata(self, mm: mmap.mmap) -> Tuple[int, int]:
  """
  Locate metadata section in the GGUF file.
  
  Args:
   mm: Memory-mapped file.
   
  Returns:
   Tuple containing offset and size of metadata section.
   
  Raises:
   GGUFFormatError: If metadata section cannot be located.
  """
  # Parse GGUF header to locate metadata
  # This is a simplified implementation - real GGUF parsing would be more complex
  version = struct.unpack('<I', mm[4:8])[0]
  
  if version == 1:
   metadata_offset = 8
   metadata_size_offset = 8
  elif version in (2, 3):
   metadata_offset = 12
   metadata_size_offset = 8
  else:
   raise GGUFFormatError(f"Unsupported GGUF version: {version}")
   
  metadata_size = struct.unpack('<I', mm[metadata_size_offset:metadata_size_offset+4])[0]
  
  logger.debug(f"Metadata located at offset {metadata_offset}, size {metadata_size} bytes")
  return metadata_offset, metadata_size

 def _parse_metadata(self, mm: mmap.mmap, offset: int, size: int) -> Dict[str, Any]:
  """
  Parse metadata from the GGUF file.
  
  Args:
   mm: Memory-mapped file.
   offset: Start offset of metadata section.
   size: Size of metadata section.
   
  Returns:
   Dict containing parsed metadata.
   
  Raises:
   DataExtractionError: If metadata parsing fails.
  """
  # Implementation depends on specific GGUF format details
  # This would need to be implemented based on the actual GGUF specification
  try:
   # In a real implementation, we'd parse the binary metadata format
   # For this example, let's simulate loading JSON data from this section
   metadata_bytes = mm[offset:offset + size]
   try:
    # Try parsing as JSON
    metadata = json.loads(metadata_bytes)
   except json.JSONDecodeError:
    # If not JSON, decode as a simple key-value format
    metadata = self._parse_binary_metadata(metadata_bytes)
    
   logger.debug(f"Metadata parsed successfully: {len(metadata)} entries")
   return metadata
   
  except Exception as e:
   raise DataExtractionError(f"Failed to parse metadata: {str(e)}")

 def _parse_binary_metadata(self, data: bytes) -> Dict[str, Any]:
  """
  Parse binary metadata format.
  
  Args:
   data: Binary metadata.
   
  Returns:
   Dict containing parsed metadata.
  """
  # Placeholder for binary metadata parsing
  # In a real implementation, this would parse the specific binary format
  # used by GGUF files
  
  # Simple parsing example (not actual GGUF format):
  result = {}
  offset = 0
  
  # Read number of key-value pairs
  kv_count = struct.unpack('<I', data[offset:offset+4])[0]
  offset += 4
  
  for _ in range(kv_count):
   # Read key length
   key_len = struct.unpack('<I', data[offset:offset+4])[0]
   offset += 4
   
   # Read key
   key = data[offset:offset+key_len].decode('utf-8')
   offset += key_len
   
   # Read value type
   value_type = data[offset]
   offset += 1
   
   # Read value based on type
   if value_type == GGUFValueType.INT32.value:
    value = struct.unpack('<i', data[offset:offset+4])[0]
    offset += 4
   elif value_type == GGUFValueType.FLOAT32.value:
    value = struct.unpack('<f', data[offset:offset+4])[0]
    offset += 4
   elif value_type == GGUFValueType.STRING.value:
    str_len = struct.unpack('<I', data[offset:offset+4])[0]
    offset += 4
    value = data[offset:offset+str_len].decode('utf-8')
    offset += str_len
   else:
    # Skip unknown types
    offset += 4
    value = None
    
   result[key] = value
   
  return result

 def _extract_binary_data(self, mm: mmap.mmap, offset: int) -> Dict[str, Any]:
  """
  Extract binary tensor data from the GGUF file.
  
  Args:
   mm: Memory-mapped file.
   offset: Start offset of binary data section.
   
  Returns:
   Dict containing extracted binary data.
   
  Raises:
   DataExtractionError: If binary data extraction fails.
  """
  # This would extract model weights, architecture parameters, etc.
  # Implementation depends on specific GGUF format details
  
  try:
   # Simplified implementation
   binary_data = {
    "arch_layers": struct.unpack('<I', mm[offset:offset+4])[0],
    "arch_hidden_size": struct.unpack('<I', mm[offset+4:offset+8])[0],
    "arch_attention_heads": struct.unpack('<I', mm[offset+8:offset+12])[0],
    "arch_vocab_size": struct.unpack('<I', mm[offset+12:offset+16])[0],
    "arch_ffn_dim": struct.unpack('<I', mm[offset+16:offset+20])[0],
    "arch_context_length": struct.unpack('<I', mm[offset+20:offset+24])[0],
   }
   
   # For binary data that needs further processing
   binary_data["tensor_count"] = struct.unpack('<I', mm[offset+24:offset+28])[0]
   binary_data["tensor_offsets"] = {}
   
   tensor_count = binary_data["tensor_count"]
   for i in range(tensor_count):
    tensor_offset = offset + 28 + (i * 12)  # Each tensor entry is 12 bytes
    tensor_name_offset = struct.unpack('<I', mm[tensor_offset:tensor_offset+4])[0]
    tensor_data_offset = struct.unpack('<Q', mm[tensor_offset+4:tensor_offset+12])[0]
    
    # Get tensor name
    name_len = struct.unpack('<I', mm[tensor_name_offset:tensor_name_offset+4])[0]
    tensor_name = mm[tensor_name_offset+4:tensor_name_offset+4+name_len].decode('utf-8')
    
    binary_data["tensor_offsets"][tensor_name] = tensor_data_offset
    
   logger.debug(f"Extracted binary data: {len(binary_data)} entries")
   return binary_data
   
  except Exception as e:
   raise DataExtractionError(f"Failed to extract binary data: {str(e)}")

 def _validate_required_fields(self) -> None:
  """
  Validate that all required fields are present and of the correct type.
  
  Raises:
   MetadataValidationError: If validation fails.
  """
  missing = []
  for field, field_type in self.REQUIRED_FIELDS.items():
   if field not in self.raw_metadata:
    missing.append(field)
   elif not isinstance(self.raw_metadata[field], field_type):
    self.errors.append(
     f"Field '{field}' is not of type {field_type.__name__}."
    )
    
  if missing:
   error_msg = f"Missing required metadata fields: {', '.join(missing)}"
   logger.error(error_msg)
   raise MetadataValidationError(error_msg)
   
  if self.errors:
   error_msg = " ".join(self.errors)
   logger.error(error_msg)
   raise MetadataValidationError(error_msg)

 def _extract_model_name(self) -> str:
  """
  Extract and validate model name.
  
  Returns:
   str: The validated model name.
   
  Raises:
   MetadataValidationError: If validation fails.
  """
  name = self.raw_metadata["model_name"].strip()
  
  if not name:
   raise MetadataValidationError("Model name cannot be empty.")
   
  # Additional validation
  if len(name) > 255:
   raise MetadataValidationError("Model name cannot exceed 255 characters.")
   
  # Check for valid model name format
  if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_\-\.]+$', name):
   raise MetadataValidationError(
    "Model name must start with alphanumeric character and contain only "
    "alphanumeric characters, underscores, hyphens, and dots."
   )
   
  logger.debug(f"Extracted model name: {name}")
  return name

 def _extract_architecture(self) -> Dict[str, Any]:
  """
  Extract and validate model architecture information.
  
  Returns:
   Dict[str, Any]: The validated architecture information.
   
  Raises:
   MetadataValidationError: If validation fails.
  """
  arch = self.raw_metadata["architecture"]
  
  # Verify required fields
  for field in self.ARCH_REQUIRED_FIELDS:
   if field not in arch:
    raise MetadataValidationError(f"Architecture missing required field: {field}")
    
   # Type validation
   expected_type = self.ARCH_REQUIRED_FIELDS[field]
   if not isinstance(arch[field], expected_type):
    raise MetadataValidationError(
     f"Architecture field '{field}' must be of type {expected_type.__name__}"
    )
  
  # Validate numeric fields
  numeric_fields = ["layers", "hidden_size", "attention_heads", "vocab_size", 
       "ffn_dim", "context_length"]
  for field in numeric_fields:
   if arch[field] <= 0:
    raise MetadataValidationError(f"Architecture '{field}' must be a positive value.")
  
  # Validate relationships between fields
  if arch["attention_heads"] > arch["hidden_size"]:
   raise MetadataValidationError(
    "Number of attention heads cannot exceed hidden size."
   )
   
  # Validate that hidden_size is divisible by attention_heads
  if arch["hidden_size"] % arch["attention_heads"] != 0:
   raise MetadataValidationError(
    "Hidden size must be divisible by the number of attention heads."
   )
   
  # Validate activation function
  valid_activations = {"relu", "gelu", "silu", "swish", "mish", "leaky_relu"}
  if arch["activation_function"].lower() not in valid_activations:
   raise MetadataValidationError(
    f"Unsupported activation function. Must be one of: {', '.join(valid_activations)}"
   )
  
  logger.debug(f"Extracted architecture with {arch['layers']} layers, "
     f"{arch['hidden_size']} hidden size")
  return arch

 def _extract_vocabulary(self) -> List[str]:
  """
  Extract and validate vocabulary information.
  
  Returns:
   List[str]: The validated vocabulary list.
   
  Raises:
   MetadataValidationError: If validation fails.
  """
  vocab = self.raw_metadata["vocabulary"]
  
  if not vocab:
   raise MetadataValidationError("Vocabulary must be a non-empty list.")
   
  if not all(isinstance(token, str) for token in vocab):
   raise MetadataValidationError("All vocabulary entries must be strings.")
  
  # Check for empty tokens
  empty_tokens = [i for i, token in enumerate(vocab) if not token]
  if empty_tokens:
   raise MetadataValidationError(
    f"Empty tokens found at indices: {empty_tokens}"
   )
   
  # Check vocabulary size against architecture
  arch = self.raw_metadata["architecture"]
  if "vocab_size" in arch and len(vocab) != arch["vocab_size"]:
   raise MetadataValidationError(
    f"Vocabulary size ({len(vocab)}) does not match architecture "
    f"vocab_size ({arch['vocab_size']})."
   )
   
  # Check for duplicates
  unique_tokens = set()
  duplicates = []
  for token in vocab:
   if token in unique_tokens:
    duplicates.append(token)
   unique_tokens.add(token)
   
  if duplicates:
   logger.warning(f"Vocabulary contains {len(duplicates)} duplicate tokens.")
  
  logger.debug(f"Extracted vocabulary of size {len(vocab)}")
  return vocab

 def _extract_provenance(self) -> Dict[str, Any]:
  """
  Extract and validate model provenance information.
  
  Returns:
   Dict[str, Any]: The validated provenance information.
   
  Raises:
   MetadataValidationError: If validation fails.
  """
  provenance = self.raw_metadata["provenance"]
  
  # Verify required fields
  for field in self.PROV_REQUIRED_FIELDS:
   if field not in provenance:
    raise MetadataValidationError(f"Provenance missing required field: {field}")
    
   # Check for empty values
   if field != "authors" and field != "training_data" and field != "parameters" and not provenance[field]:
    raise MetadataValidationError(f"Provenance '{field}' cannot be empty.")
  
  # Validate authors
  if not isinstance(provenance["authors"], list) or not provenance["authors"]:
   raise MetadataValidationError("Provenance 'authors' must be a non-empty list.")
  
  for i, author in enumerate(provenance["authors"]):
   if not isinstance(author, str) or not author:
    raise MetadataValidationError(f"Author at index {i} must be a non-empty string.")
  
  # Validate training data
  if not isinstance(provenance["training_data"], list):
   raise MetadataValidationError("Provenance 'training_data' must be a list.")
   
  for i, dataset in enumerate(provenance["training_data"]):
   if not isinstance(dataset, dict):
    raise MetadataValidationError(f"Dataset at index {i} must be a dictionary.")
   
   if "name" not in dataset or not dataset["name"]:
    raise MetadataValidationError(f"Dataset at index {i} missing 'name' field.")
  
  # Validate parameters
  if not isinstance(provenance["parameters"], dict):
   raise MetadataValidationError("Provenance 'parameters' must be a dictionary.")
   
  # Validate creation date
  try:
   datetime.datetime.fromisoformat(provenance["creation_date"])
  except ValueError:
   raise MetadataValidationError(
    "Provenance 'creation_date' must be a valid ISO 8601 date string."
   )
  
  logger.debug(f"Extracted provenance with {len(provenance['authors'])} authors")
  return provenance

 def _extract_sha256(self) -> str:
  """
  Extract and validate SHA-256 hash.
  
  Returns:
   str: The validated SHA-256 hash.
   
  Raises:
   MetadataValidationError: If validation fails.
  """
  sha = self.raw_metadata["sha256"]
  
  # SHA-256 validation
  if not isinstance(sha, str) or len(sha) != 64:
   raise MetadataValidationError("SHA-256 must be a 64-character hexadecimal string.")
   
  try:
   int(sha, 16)
  except ValueError:
   raise MetadataValidationError("SHA-256 must be a valid hexadecimal string.")
   
  # Check if file hash matches metadata hash (if available)
  if self.file_hash and sha != self.file_hash:
   logger.warning(
    f"SHA-256 in metadata ({sha}) does not match computed file hash ({self.file_hash})"
   )
  
  logger.debug(f"Extracted SHA-256: {sha}")
  return sha

 def _extract_created_at(self) -> str:
  """
  Extract and validate creation timestamp.
  
  Returns:
   str: The validated creation timestamp.
   
  Raises:
   MetadataValidationError: If validation fails.
  """
  created_at = self.raw_metadata["created_at"]
  
  # Validate ISO 8601 format
  try:
   datetime.datetime.fromisoformat(created_at)
  except ValueError:
   raise MetadataValidationError("created_at must be a valid ISO 8601 datetime string.")
   
  # Additional validation
  if not created_at.endswith('Z') and '+' not in created_at and '-' not in created_at[10:]:
   logger.warning("created_at should include timezone information")
  
  # Validate that creation date is not in the future
  creation_dt = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
  now = datetime.datetime.now(datetime.timezone.utc)
  if creation_dt > now:
   raise MetadataValidationError("created_at timestamp cannot be in the future.")
   
  logger.debug(f"Extracted created_at: {created_at}")
  return created_at

 def _extract_version(self) -> str:
  """
  Extract and validate version string.
  
  Returns:
   str: The validated version string.
   
  Raises:
   MetadataValidationError: If validation fails.
  """
  version = self.raw_metadata["version"]
  
  if not isinstance(version, str) or not version:
   raise MetadataValidationError("Version must be a non-empty string.")
   
  # Validate semantic versioning format
  if not re.match(r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$', version):
   raise MetadataValidationError(
    "Version must follow semantic versioning format (e.g. 1.0.0)."
   )
  
  logger.debug(f"Extracted version: {version}")
  return version

 def _validate_metadata(self) -> None:
  """
  Perform final validation on the entire metadata structure.
  
  This method performs cross-field validations and consistency checks
  on the extracted metadata.
  
  Raises:
   MetadataValidationError: If validation fails.
  """
  # Vocabulary validation checks
  vocab = self.metadata.get("vocabulary", [])
  if len(vocab) != len(set(vocab)):
   logger.warning("Vocabulary contains duplicate tokens")
   
  # Ensure numeric values are consistent
  arch = self.metadata.get("architecture", {})
  if "hidden_size" in arch and "attention_heads" in arch:
   if arch["hidden_size"] % arch["attention_heads"] != 0:
    logger.warning("Hidden size is not divisible by number of attention heads")
    
  # Validate model name for forbidden filesystem characters
  forbidden_chars = r'\/:*?"<>|'
  if any(c in self.metadata["model_name"] for c in forbidden_chars):
   raise MetadataValidationError(
    f"Model name contains forbidden filesystem characters: {forbidden_chars}"
   )
   
  # Validate SHA-256 consistency
  if self.file_hash and self.metadata["sha256"] != self.file_hash:
   logger.warning("SHA-256 in metadata doesn't match computed file hash")
   
  # Check provenance creation date consistency with metadata created_at
  prov = self.metadata.get("provenance", {})
  if "creation_date" in prov and self.metadata.get("created_at", "") != prov["creation_date"]:
   logger.warning("Inconsistent creation dates between metadata and provenance")
  
  logger.info("Metadata validation passed")

 def get_metadata(self) -> Dict[str, Any]:
  """
  Get the validated metadata.
  
  Returns:
   Dict[str, Any]: The validated metadata dictionary.
   
  Raises:
   MetadataValidationError: If metadata has not been extracted yet.
  """
  if not self.metadata:
   raise MetadataValidationError("Metadata has not been extracted yet.")
  return self.metadata

 def get_tensor_info(self) -> Dict[str, Dict[str, Any]]:
  """
  Get information about tensor data in the model.
  
  Returns:
   Dict[str, Dict[str, Any]]: Information about each tensor.
   
  Raises:
   DataExtractionError: If tensor information is not available.
  """
  if not self.binary_data or "tensor_offsets" not in self.binary_data:
   raise DataExtractionError("Tensor information not available")
   
  tensor_info = {}
  for tensor_name, offset in self.binary_data["tensor_offsets"].items():
   tensor_info[tensor_name] = {
    "offset": offset,
    "name": tensor_name,
   }
   
  return tensor_info

 def get_vocabulary_statistics(self) -> Dict[str, Any]:
  """
  Get statistical information about the vocabulary.
  
  Returns:
   Dict[str, Any]: Vocabulary statistics.
   
  Raises:
   MetadataValidationError: If vocabulary is not available.
  """
  vocab = self.metadata.get("vocabulary")
  if not vocab:
   raise MetadataValidationError("Vocabulary not available")
   
  # Calculate statistics
  token_lengths = [len(token) for token in vocab]
  stats = {
   "count": len(vocab),
   "unique_count": len(set(vocab)),
   "min_length": min(token_lengths),
   "max_length": max(token_lengths),
   "avg_length": sum(token_lengths) / len(token_lengths),
   "total_chars": sum(token_lengths),
  }
  
  # Count special tokens
  special_prefixes = ["<", "[", "]", "{", "}", "#", "$", "@"]
  stats["special_tokens"] = sum(1 for t in vocab if any(t.startswith(p) for p in special_prefixes))
  
  return stats

 def write_metadata_json(self, output_path: Union[str, Path]) -> None:
  """
  Write extracted metadata to a JSON file.
  
  Args:
   output_path: Path to write the JSON file.
   
  Raises:
   MetadataValidationError: If metadata has not been extracted yet.
  """
  if not self.metadata:
   raise MetadataValidationError("Metadata has not been extracted yet.")
   
  output_path = Path(output_path)
  with open(output_path, 'w', encoding='utf-8') as f:
   json.dump(self.metadata, f, indent=2, ensure_ascii=False)
   
  logger.info(f"Metadata written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="SVELTE Metadata Extractor CLI")
    parser.add_argument('model', type=str, help='Path to GGUF model file')
    args = parser.parse_args()
    from src.tensor_analysis.gguf_parser import GGUFParser
    gguf = GGUFParser(args.model)
    gguf.parse()
    extractor = MetadataExtractor(gguf.get_metadata())
    extractor.extract()
    metadata = extractor.get_metadata()
    import json
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
