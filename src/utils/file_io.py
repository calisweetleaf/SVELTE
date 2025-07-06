# src/utils/file_io.py
"""
File I/O utilities for SVELTE Framework.
Handles reading GGUF, GGML, and related model files, as well as output serialization.
Provides robust, production-grade error handling and advanced serialization features.
"""
import os
import json
import struct
import shutil
import tempfile
from typing import Any, Dict, Optional, Callable, IO, Union, List

class FileIOException(Exception):
 """Custom exception for file I/O errors."""
 def __init__(self, message: str, original_exception: Optional[Exception] = None):
  super().__init__(message)
  self.original_exception = original_exception

def read_binary_file(filepath: str) -> bytes:
 """
 Read a binary file and return its contents as bytes.
 Raises FileIOException on failure.
 """
 if not os.path.isfile(filepath):
  raise FileIOException(f"File not found: {filepath}")
 try:
  with open(filepath, 'rb') as f:
   data = f.read()
  return data
 except Exception as e:
  raise FileIOException(f"Failed to read file {filepath}: {e}", e)

def write_json(data: Any, filepath: str, atomic: bool = True, sort_keys: bool = False):
 """
 Write data as JSON to a file.
 Supports atomic write to prevent data corruption.
 Raises FileIOException on failure.
 """
 dirpath = os.path.dirname(os.path.abspath(filepath))
 if not os.path.isdir(dirpath):
  os.makedirs(dirpath, exist_ok=True)
 try:
  if atomic:
   with tempfile.NamedTemporaryFile('w', dir=dirpath, delete=False, encoding='utf-8') as tf:
    json.dump(data, tf, indent=2, sort_keys=sort_keys, ensure_ascii=False)
    tempname = tf.name
   os.replace(tempname, filepath)
  else:
   with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, sort_keys=sort_keys, ensure_ascii=False)
 except Exception as e:
  raise FileIOException(f"Failed to write JSON to {filepath}: {e}", e)

def read_json(filepath: str, object_hook: Optional[Callable[[Dict], Any]] = None) -> Any:
 """
 Read JSON data from a file.
 Optionally uses an object_hook for custom decoding.
 Raises FileIOException on failure.
 """
 if not os.path.isfile(filepath):
  raise FileIOException(f"File not found: {filepath}")
 try:
  with open(filepath, 'r', encoding='utf-8') as f:
   if object_hook:
    return json.load(f, object_hook=object_hook)
   return json.load(f)
 except Exception as e:
  raise FileIOException(f"Failed to read JSON from {filepath}: {e}", e)

def ensure_dir(path: str, mode: int = 0o755):
 """
 Ensure a directory exists. Creates it if it does not exist.
 Raises FileIOException on failure.
 """
 try:
  os.makedirs(path, mode=mode, exist_ok=True)
 except Exception as e:
  raise FileIOException(f"Failed to create directory {path}: {e}", e)

def copy_file(src: str, dst: str, overwrite: bool = False):
 """
 Copy a file from src to dst.
 If overwrite is False and dst exists, raises FileIOException.
 """
 if not os.path.isfile(src):
  raise FileIOException(f"Source file not found: {src}")
 if os.path.exists(dst) and not overwrite:
  raise FileIOException(f"Destination file already exists: {dst}")
 try:
  shutil.copy2(src, dst)
 except Exception as e:
  raise FileIOException(f"Failed to copy file from {src} to {dst}: {e}", e)

def read_lines(filepath: str, strip: bool = True, encoding: str = 'utf-8') -> List[str]:
 """
 Read all lines from a text file.
 Optionally strips whitespace from each line.
 """
 if not os.path.isfile(filepath):
  raise FileIOException(f"File not found: {filepath}")
 try:
  with open(filepath, 'r', encoding=encoding) as f:
   lines = f.readlines()
  if strip:
   lines = [line.strip() for line in lines]
  return lines
 except Exception as e:
  raise FileIOException(f"Failed to read lines from {filepath}: {e}", e)

def write_lines(lines: List[str], filepath: str, atomic: bool = True, encoding: str = 'utf-8'):
 """
 Write a list of lines to a text file.
 Supports atomic write.
 """
 dirpath = os.path.dirname(os.path.abspath(filepath))
 if not os.path.isdir(dirpath):
  os.makedirs(dirpath, exist_ok=True)
 try:
  if atomic:
   with tempfile.NamedTemporaryFile('w', dir=dirpath, delete=False, encoding=encoding) as tf:
    for line in lines:
     tf.write(f"{line}\n")
    tempname = tf.name
   os.replace(tempname, filepath)
  else:
   with open(filepath, 'w', encoding=encoding) as f:
    for line in lines:
     f.write(f"{line}\n")
 except Exception as e:
  raise FileIOException(f"Failed to write lines to {filepath}: {e}", e)

def read_struct(filepath: str, fmt: str, offset: int = 0) -> tuple:
 """
 Read and unpack a struct from a binary file at a given offset.
 fmt: struct format string.
 Returns a tuple of unpacked values.
 """
 if not os.path.isfile(filepath):
  raise FileIOException(f"File not found: {filepath}")
 try:
  with open(filepath, 'rb') as f:
   f.seek(offset)
   size = struct.calcsize(fmt)
   data = f.read(size)
   if len(data) != size:
    raise FileIOException(f"Could not read enough bytes for struct '{fmt}' from {filepath}")
   return struct.unpack(fmt, data)
 except Exception as e:
  raise FileIOException(f"Failed to read struct from {filepath}: {e}", e)

def safe_remove(filepath: str):
 """
 Remove a file if it exists.
 Raises FileIOException on failure.
 """
 try:
  if os.path.isfile(filepath):
   os.remove(filepath)
 except Exception as e:
  raise FileIOException(f"Failed to remove file {filepath}: {e}", e)

def list_files(directory: str, pattern: Optional[str] = None, recursive: bool = False) -> List[str]:
 """
 List files in a directory.
 If pattern is provided, only files matching the pattern are returned.
 If recursive is True, traverses subdirectories.
 """
 if not os.path.isdir(directory):
  raise FileIOException(f"Directory not found: {directory}")
 result = []
 try:
  if recursive:
   for root, _, files in os.walk(directory):
    for file in files:
     if pattern is None or pattern in file:
      result.append(os.path.join(root, file))
  else:
   for file in os.listdir(directory):
    full_path = os.path.join(directory, file)
    if os.path.isfile(full_path):
     if pattern is None or pattern in file:
      result.append(full_path)
  return result
 except Exception as e:
  raise FileIOException(f"Failed to list files in {directory}: {e}", e)

def file_size(filepath: str) -> int:
 """
 Return the size of a file in bytes.
 Raises FileIOException if file does not exist.
 """
 if not os.path.isfile(filepath):
  raise FileIOException(f"File not found: {filepath}")
 try:
  return os.path.getsize(filepath)
 except Exception as e:
  raise FileIOException(f"Failed to get file size for {filepath}: {e}", e)

def is_json_file(filepath: str) -> bool:
 """
 Check if a file is a valid JSON file.
 Returns True if valid, False otherwise.
 """
 try:
  with open(filepath, 'r', encoding='utf-8') as f:
   json.load(f)
  return True
 except Exception:
  return False

def move_file(src: str, dst: str, overwrite: bool = False):
 """
 Move a file from src to dst.
 If overwrite is False and dst exists, raises FileIOException.
 """
 if not os.path.isfile(src):
  raise FileIOException(f"Source file not found: {src}")
 if os.path.exists(dst) and not overwrite:
  raise FileIOException(f"Destination file already exists: {dst}")
 try:
  shutil.move(src, dst)
 except Exception as e:
  raise FileIOException(f"Failed to move file from {src} to {dst}: {e}", e)
