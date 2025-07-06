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
import hashlib
import gzip
import pickle
import csv
import logging
import mmap
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable, IO, Union, List, Tuple, Iterator, BinaryIO
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class FileIOException(Exception):
    """Custom exception for file I/O errors."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception

class FileChecksum:
    """Utility class for file integrity verification."""
    
    @staticmethod
    def calculate_md5(filepath: str, chunk_size: int = 8192) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            raise FileIOException(f"Failed to calculate MD5 for {filepath}: {e}", e)
    
    @staticmethod
    def calculate_sha256(filepath: str, chunk_size: int = 8192) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            raise FileIOException(f"Failed to calculate SHA256 for {filepath}: {e}", e)
    
    @staticmethod
    def verify_checksum(filepath: str, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """Verify file integrity using checksum."""
        try:
            if algorithm.lower() == 'md5':
                actual_hash = FileChecksum.calculate_md5(filepath)
            elif algorithm.lower() == 'sha256':
                actual_hash = FileChecksum.calculate_sha256(filepath)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            return actual_hash.lower() == expected_hash.lower()
        except Exception as e:
            raise FileIOException(f"Failed to verify checksum for {filepath}: {e}", e)

def read_binary_file(filepath: str, max_size: Optional[int] = None) -> bytes:
    """
    Read a binary file and return its contents as bytes.
    
    Args:
        filepath: Path to the file
        max_size: Maximum file size to read (None for no limit)
        
    Raises:
        FileIOException: On failure
    """
    if not os.path.isfile(filepath):
        raise FileIOException(f"File not found: {filepath}")
    
    # Check file size if limit specified
    if max_size:
        file_size = os.path.getsize(filepath)
        if file_size > max_size:
            raise FileIOException(f"File too large: {file_size} bytes (max: {max_size})")
    
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        logger.debug(f"Successfully read {len(data)} bytes from {filepath}")
        return data
    except Exception as e:
        raise FileIOException(f"Failed to read file {filepath}: {e}", e)

def read_binary_chunk(filepath: str, offset: int, size: int) -> bytes:
    """
    Read a specific chunk of bytes from a binary file.
    
    Args:
        filepath: Path to the file
        offset: Byte offset to start reading
        size: Number of bytes to read
        
    Returns:
        bytes: The requested chunk
    """
    if not os.path.isfile(filepath):
        raise FileIOException(f"File not found: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            f.seek(offset)
            data = f.read(size)
            if len(data) != size:
                raise FileIOException(f"Could not read {size} bytes from offset {offset}")
            return data
    except Exception as e:
        raise FileIOException(f"Failed to read chunk from {filepath}: {e}", e)

def write_binary_file(data: bytes, filepath: str, atomic: bool = True) -> None:
    """
    Write binary data to a file with atomic write support.
    
    Args:
        data: Binary data to write
        filepath: Destination file path
        atomic: Use atomic write to prevent corruption
    """
    dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    
    try:
        if atomic:
            with tempfile.NamedTemporaryFile(dir=dirpath, delete=False) as tf:
                tf.write(data)
                tempname = tf.name
            os.replace(tempname, filepath)
        else:
            with open(filepath, 'wb') as f:
                f.write(data)
        logger.debug(f"Successfully wrote {len(data)} bytes to {filepath}")
    except Exception as e:
        raise FileIOException(f"Failed to write binary file {filepath}: {e}", e)

@contextmanager
def memory_map_file(filepath: str, mode: str = 'r') -> Iterator[mmap.mmap]:
    """
    Context manager for memory-mapped file access.
    
    Args:
        filepath: Path to the file
        mode: Access mode ('r' for read, 'w' for write)
        
    Yields:
        mmap.mmap: Memory-mapped file object
    """
    if not os.path.isfile(filepath):
        raise FileIOException(f"File not found: {filepath}")
    
    try:
        with open(filepath, 'rb' if mode == 'r' else 'r+b') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ if mode == 'r' else mmap.ACCESS_WRITE) as mm:
                yield mm
    except Exception as e:
        raise FileIOException(f"Failed to memory map file {filepath}: {e}", e)

def stream_binary_file(filepath: str, chunk_size: int = 8192) -> Iterator[bytes]:
    """
    Stream a binary file in chunks.
    
    Args:
        filepath: Path to the file
        chunk_size: Size of each chunk in bytes
        
    Yields:
        bytes: File chunks
    """
    if not os.path.isfile(filepath):
        raise FileIOException(f"File not found: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        raise FileIOException(f"Failed to stream file {filepath}: {e}", e)

def write_json(data: Any, filepath: str, atomic: bool = True, sort_keys: bool = False, 
               indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    Write data as JSON to a file with enhanced options.
    
    Args:
        data: Data to serialize
        filepath: Destination file path
        atomic: Use atomic write to prevent corruption
        sort_keys: Sort dictionary keys
        indent: JSON indentation level
        ensure_ascii: Ensure ASCII output
    """
    dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    
    try:
        json_str = json.dumps(data, indent=indent, sort_keys=sort_keys, 
                             ensure_ascii=ensure_ascii, default=_json_serializer)
        
        if atomic:
            with tempfile.NamedTemporaryFile('w', dir=dirpath, delete=False, 
                                           encoding='utf-8', suffix='.tmp') as tf:
                tf.write(json_str)
                tempname = tf.name
            os.replace(tempname, filepath)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        logger.debug(f"Successfully wrote JSON to {filepath}")
    except Exception as e:
        raise FileIOException(f"Failed to write JSON to {filepath}: {e}", e)

def read_json(filepath: str, object_hook: Optional[Callable[[Dict], Any]] = None) -> Any:
    """
    Read JSON data from a file with comprehensive error handling.
    
    Args:
        filepath: Path to JSON file
        object_hook: Custom object decoder
        
    Returns:
        Parsed JSON data
    """
    if not os.path.isfile(filepath):
        raise FileIOException(f"File not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                raise FileIOException(f"Empty JSON file: {filepath}")
            
            if object_hook:
                return json.loads(content, object_hook=object_hook)
            return json.loads(content)
    except json.JSONDecodeError as e:
        raise FileIOException(f"Invalid JSON in {filepath}: {e}", e)
    except Exception as e:
        raise FileIOException(f"Failed to read JSON from {filepath}: {e}", e)

def write_compressed_json(data: Any, filepath: str, compression_level: int = 6) -> None:
    """
    Write JSON data with gzip compression.
    
    Args:
        data: Data to serialize
        filepath: Destination file path
        compression_level: Compression level (1-9)
    """
    dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    
    try:
        json_str = json.dumps(data, indent=2, sort_keys=True, 
                             ensure_ascii=False, default=_json_serializer)
        
        with gzip.open(filepath, 'wt', encoding='utf-8', compresslevel=compression_level) as f:
            f.write(json_str)
        
        logger.debug(f"Successfully wrote compressed JSON to {filepath}")
    except Exception as e:
        raise FileIOException(f"Failed to write compressed JSON to {filepath}: {e}", e)

def read_compressed_json(filepath: str) -> Any:
    """
    Read gzip-compressed JSON data.
    
    Args:
        filepath: Path to compressed JSON file
        
    Returns:
        Parsed JSON data
    """
    if not os.path.isfile(filepath):
        raise FileIOException(f"File not found: {filepath}")
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise FileIOException(f"Failed to read compressed JSON from {filepath}: {e}", e)

def write_pickle(data: Any, filepath: str, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """
    Serialize data using pickle.
    
    Args:
        data: Data to serialize
        filepath: Destination file path
        protocol: Pickle protocol version
    """
    dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)
        logger.debug(f"Successfully wrote pickle to {filepath}")
    except Exception as e:
        raise FileIOException(f"Failed to write pickle to {filepath}: {e}", e)

def read_pickle(filepath: str) -> Any:
    """
    Deserialize data from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Deserialized data
    """
    if not os.path.isfile(filepath):
        raise FileIOException(f"File not found: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise FileIOException(f"Failed to read pickle from {filepath}: {e}", e)

def write_csv(data: List[List[Any]], filepath: str, headers: Optional[List[str]] = None,
              delimiter: str = ',', quoting: int = csv.QUOTE_MINIMAL) -> None:
    """
    Write data to CSV file.
    
    Args:
        data: List of rows to write
        filepath: Destination file path
        headers: Optional column headers
        delimiter: CSV delimiter
        quoting: CSV quoting style
    """
    dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=delimiter, quoting=quoting)
            if headers:
                writer.writerow(headers)
            writer.writerows(data)
        logger.debug(f"Successfully wrote CSV to {filepath}")
    except Exception as e:
        raise FileIOException(f"Failed to write CSV to {filepath}: {e}", e)

def read_csv(filepath: str, headers: bool = True, delimiter: str = ',') -> Union[List[List[str]], Tuple[List[str], List[List[str]]]]:
    """
    Read CSV file.
    
    Args:
        filepath: Path to CSV file
        headers: Whether first row contains headers
        delimiter: CSV delimiter
        
    Returns:
        CSV data (with headers if requested)
    """
    if not os.path.isfile(filepath):
        raise FileIOException(f"File not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
            
            if headers and rows:
                return rows[0], rows[1:]
            return rows
    except Exception as e:
        raise FileIOException(f"Failed to read CSV from {filepath}: {e}", e)

def write_numpy_array(array: np.ndarray, filepath: str, compressed: bool = True) -> None:
    """
    Write numpy array to file.
    
    Args:
        array: NumPy array to save
        filepath: Destination file path
        compressed: Use compression
    """
    dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    
    try:
        if compressed:
            np.savez_compressed(filepath, array=array)
        else:
            np.save(filepath, array)
        logger.debug(f"Successfully wrote numpy array to {filepath}")
    except Exception as e:
        raise FileIOException(f"Failed to write numpy array to {filepath}: {e}", e)

def read_numpy_array(filepath: str) -> np.ndarray:
    """
    Read numpy array from file.
    
    Args:
        filepath: Path to numpy file
        
    Returns:
        Loaded numpy array
    """
    if not os.path.isfile(filepath):
        raise FileIOException(f"File not found: {filepath}")
    
    try:
        if filepath.endswith('.npz'):
            with np.load(filepath) as data:
                if 'array' in data:
                    return data['array']
                else:
                    # Return first array if no 'array' key
                    return data[list(data.keys())[0]]
        else:
            return np.load(filepath)
    except Exception as e:
        raise FileIOException(f"Failed to read numpy array from {filepath}: {e}", e)

def ensure_dir(path: str, mode: int = 0o755) -> None:
    """
    Ensure a directory exists with comprehensive validation.
    
    Args:
        path: Directory path to create
        mode: Directory permissions
    """
    try:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True, mode=mode)
        logger.debug(f"Directory ensured: {path}")
    except Exception as e:
        raise FileIOException(f"Failed to create directory {path}: {e}", e)

def copy_file(src: str, dst: str, overwrite: bool = False, preserve_metadata: bool = True) -> None:
    """
    Copy a file with enhanced options.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Allow overwriting existing files
        preserve_metadata: Preserve file metadata
    """
    if not os.path.isfile(src):
        raise FileIOException(f"Source file not found: {src}")
    
    if os.path.exists(dst) and not overwrite:
        raise FileIOException(f"Destination file already exists: {dst}")
    
    # Ensure destination directory exists
    dst_dir = os.path.dirname(dst)
    if dst_dir:
        ensure_dir(dst_dir)
    
    try:
        if preserve_metadata:
            shutil.copy2(src, dst)
        else:
            shutil.copy(src, dst)
        logger.debug(f"Successfully copied {src} to {dst}")
    except Exception as e:
        raise FileIOException(f"Failed to copy file from {src} to {dst}: {e}", e)

def move_file(src: str, dst: str, overwrite: bool = False) -> None:
    """
    Move a file with comprehensive validation.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Allow overwriting existing files
    """
    if not os.path.isfile(src):
        raise FileIOException(f"Source file not found: {src}")
    
    if os.path.exists(dst) and not overwrite:
        raise FileIOException(f"Destination file already exists: {dst}")
    
    # Ensure destination directory exists
    dst_dir = os.path.dirname(dst)
    if dst_dir:
        ensure_dir(dst_dir)
    
    try:
        shutil.move(src, dst)
        logger.debug(f"Successfully moved {src} to {dst}")
    except Exception as e:
        raise FileIOException(f"Failed to move file from {src} to {dst}: {e}", e)

def read_lines(filepath: str, strip: bool = True, encoding: str = 'utf-8', 
               skip_empty: bool = False) -> List[str]:
    """
    Read all lines from a text file with enhanced options.
    
    Args:
        filepath: Path to text file
        strip: Strip whitespace from lines
        encoding: File encoding
        skip_empty: Skip empty lines
        
    Returns:
        List of lines
    """
    if not os.path.isfile(filepath):
        raise FileIOException(f"File not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            lines = f.readlines()
        
        if strip:
            lines = [line.strip() for line in lines]
        
        if skip_empty:
            lines = [line for line in lines if line]
        
        return lines
    except Exception as e:
        raise FileIOException(f"Failed to read lines from {filepath}: {e}", e)

def write_lines(lines: List[str], filepath: str, atomic: bool = True, 
                encoding: str = 'utf-8', line_ending: str = '\n') -> None:
    """
    Write lines to a text file with enhanced options.
    
    Args:
        lines: List of lines to write
        filepath: Destination file path
        atomic: Use atomic write
        encoding: File encoding
        line_ending: Line ending character(s)
    """
    dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    
    try:
        content = line_ending.join(lines) + line_ending
        
        if atomic:
            with tempfile.NamedTemporaryFile('w', dir=dirpath, delete=False, 
                                           encoding=encoding, suffix='.tmp') as tf:
                tf.write(content)
                tempname = tf.name
            os.replace(tempname, filepath)
        else:
            with open(filepath, 'w', encoding=encoding) as f:
                f.write(content)
        
        logger.debug(f"Successfully wrote {len(lines)} lines to {filepath}")
    except Exception as e:
        raise FileIOException(f"Failed to write lines to {filepath}: {e}", e)

def read_struct(filepath: str, fmt: str, offset: int = 0) -> tuple:
    """
    Read and unpack a struct from a binary file at a given offset.
    
    Args:
        filepath: Path to binary file
        fmt: struct format string
        offset: Byte offset to start reading
        
    Returns:
        Tuple of unpacked values
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

def write_struct(data: tuple, fmt: str, filepath: str, offset: int = 0, mode: str = 'wb') -> None:
    """
    Pack and write a struct to a binary file.
    
    Args:
        data: Tuple of values to pack
        fmt: struct format string
        filepath: Destination file path
        offset: Byte offset to start writing
        mode: File open mode
    """
    try:
        packed_data = struct.pack(fmt, *data)
        
        if mode == 'wb':
            # Write entire file
            dirpath = os.path.dirname(os.path.abspath(filepath))
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath, exist_ok=True)
            
            with open(filepath, mode) as f:
                f.seek(offset)
                f.write(packed_data)
        else:
            # Append or modify existing file
            with open(filepath, mode) as f:
                f.seek(offset)
                f.write(packed_data)
        
        logger.debug(f"Successfully wrote struct to {filepath}")
    except Exception as e:
        raise FileIOException(f"Failed to write struct to {filepath}: {e}", e)

def safe_remove(filepath: str, missing_ok: bool = True) -> bool:
    """
    Remove a file if it exists with comprehensive error handling.
    
    Args:
        filepath: Path to file to remove
        missing_ok: Don't raise error if file doesn't exist
        
    Returns:
        True if file was removed, False if it didn't exist
    """
    try:
        if os.path.isfile(filepath):
            os.remove(filepath)
            logger.debug(f"Successfully removed {filepath}")
            return True
        elif not missing_ok:
            raise FileIOException(f"File not found: {filepath}")
        return False
    except Exception as e:
        raise FileIOException(f"Failed to remove file {filepath}: {e}", e)

def safe_remove_dir(dirpath: str, missing_ok: bool = True, recursive: bool = False) -> bool:
    """
    Remove a directory safely.
    
    Args:
        dirpath: Path to directory to remove
        missing_ok: Don't raise error if directory doesn't exist
        recursive: Remove directory tree recursively
        
    Returns:
        True if directory was removed, False if it didn't exist
    """
    try:
        if os.path.isdir(dirpath):
            if recursive:
                shutil.rmtree(dirpath)
            else:
                os.rmdir(dirpath)
            logger.debug(f"Successfully removed directory {dirpath}")
            return True
        elif not missing_ok:
            raise FileIOException(f"Directory not found: {dirpath}")
        return False
    except Exception as e:
        raise FileIOException(f"Failed to remove directory {dirpath}: {e}", e)

def list_files(directory: str, pattern: Optional[str] = None, recursive: bool = False,
               include_dirs: bool = False, sort: bool = True) -> List[str]:
    """
    List files in a directory with enhanced filtering.
    
    Args:
        directory: Directory to search
        pattern: Pattern to match (substring or glob)
        recursive: Search subdirectories
        include_dirs: Include directories in results
        sort: Sort results alphabetically
        
    Returns:
        List of matching file paths
    """
    if not os.path.isdir(directory):
        raise FileIOException(f"Directory not found: {directory}")
    
    result = []
    
    try:
        if recursive:
            for root, dirs, files in os.walk(directory):
                # Add files
                for file in files:
                    full_path = os.path.join(root, file)
                    if pattern is None or _match_pattern(file, pattern):
                        result.append(full_path)
                
                # Add directories if requested
                if include_dirs:
                    for dir_name in dirs:
                        full_path = os.path.join(root, dir_name)
                        if pattern is None or _match_pattern(dir_name, pattern):
                            result.append(full_path)
        else:
            for item in os.listdir(directory):
                full_path = os.path.join(directory, item)
                is_file = os.path.isfile(full_path)
                is_dir = os.path.isdir(full_path)
                
                if (is_file or (include_dirs and is_dir)):
                    if pattern is None or _match_pattern(item, pattern):
                        result.append(full_path)
        
        if sort:
            result.sort()
        
        return result
    except Exception as e:
        raise FileIOException(f"Failed to list files in {directory}: {e}", e)

def get_file_info(filepath: str) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        filepath: Path to file
        
    Returns:
        Dictionary with file information
    """
    if not os.path.exists(filepath):
        raise FileIOException(f"File not found: {filepath}")
    
    try:
        stat = os.stat(filepath)
        info = {
            'path': os.path.abspath(filepath),
            'name': os.path.basename(filepath),
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'accessed': datetime.fromtimestamp(stat.st_atime),
            'is_file': os.path.isfile(filepath),
            'is_dir': os.path.isdir(filepath),
            'permissions': oct(stat.st_mode)[-3:],
            'extension': os.path.splitext(filepath)[1].lower()
        }
        
        # Add file type specific info
        if info['is_file']:
            info['mime_type'] = _get_mime_type(filepath)
            if info['extension'] in ['.json', '.txt', '.py', '.md']:
                info['encoding'] = _detect_encoding(filepath)
        
        return info
    except Exception as e:
        raise FileIOException(f"Failed to get file info for {filepath}: {e}", e)

def file_size(filepath: str) -> int:
    """
    Return the size of a file in bytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in bytes
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
    
    Args:
        filepath: Path to file
        
    Returns:
        True if valid JSON, False otherwise
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError):
        return False

def is_binary_file(filepath: str, chunk_size: int = 1024) -> bool:
    """
    Check if a file is binary by examining its content.
    
    Args:
        filepath: Path to file
        chunk_size: Size of chunk to examine
        
    Returns:
        True if binary, False if text
    """
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(chunk_size)
            if b'\0' in chunk:
                return True
            
            # Check for high ratio of non-printable characters
            printable = sum(32 <= byte <= 126 for byte in chunk)
            return printable / len(chunk) < 0.7 if chunk else False
    except Exception:
        return False

def create_backup(filepath: str, backup_dir: Optional[str] = None, 
                  timestamp: bool = True) -> str:
    """
    Create a backup of a file.
    
    Args:
        filepath: Path to file to backup
        backup_dir: Directory to store backup (None for same directory)
        timestamp: Add timestamp to backup filename
        
    Returns:
        Path to backup file
    """
    if not os.path.isfile(filepath):
        raise FileIOException(f"File not found: {filepath}")
    
    try:
        base_name = os.path.basename(filepath)
        name, ext = os.path.splitext(base_name)
        
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{name}_{timestamp_str}{ext}"
        else:
            backup_name = f"{name}_backup{ext}"
        
        if backup_dir:
            ensure_dir(backup_dir)
            backup_path = os.path.join(backup_dir, backup_name)
        else:
            backup_path = os.path.join(os.path.dirname(filepath), backup_name)
        
        shutil.copy2(filepath, backup_path)
        logger.debug(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        raise FileIOException(f"Failed to create backup of {filepath}: {e}", e)

def cleanup_temp_files(directory: str, pattern: str = "*.tmp", age_hours: int = 24) -> int:
    """
    Clean up temporary files older than specified age.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        age_hours: Age threshold in hours
        
    Returns:
        Number of files removed
    """
    if not os.path.isdir(directory):
        raise FileIOException(f"Directory not found: {directory}")
    
    import time
    import fnmatch
    
    threshold = time.time() - (age_hours * 3600)
    removed_count = 0
    
    try:
        for filename in os.listdir(directory):
            if fnmatch.fnmatch(filename, pattern):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath) and os.path.getmtime(filepath) < threshold:
                    os.remove(filepath)
                    removed_count += 1
        
        logger.debug(f"Cleaned up {removed_count} temporary files from {directory}")
        return removed_count
    except Exception as e:
        raise FileIOException(f"Failed to cleanup temp files in {directory}: {e}", e)

def sync_directories(src_dir: str, dst_dir: str, delete_extra: bool = False) -> Dict[str, int]:
    """
    Synchronize two directories.
    
    Args:
        src_dir: Source directory
        dst_dir: Destination directory
        delete_extra: Delete files in destination not in source
        
    Returns:
        Dictionary with sync statistics
    """
    if not os.path.isdir(src_dir):
        raise FileIOException(f"Source directory not found: {src_dir}")
    
    ensure_dir(dst_dir)
    
    stats = {'copied': 0, 'updated': 0, 'deleted': 0, 'errors': 0}
    
    try:
        # Copy/update files from source to destination
        for root, dirs, files in os.walk(src_dir):
            rel_root = os.path.relpath(root, src_dir)
            dst_root = os.path.join(dst_dir, rel_root) if rel_root != '.' else dst_dir
            
            ensure_dir(dst_root)
            
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_root, file)
                
                try:
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
                        stats['copied'] += 1
                    elif os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                        shutil.copy2(src_file, dst_file)
                        stats['updated'] += 1
                except Exception as e:
                    logger.warning(f"Failed to sync {src_file}: {e}")
                    stats['errors'] += 1
        
        # Delete extra files if requested
        if delete_extra:
            for root, dirs, files in os.walk(dst_dir):
                rel_root = os.path.relpath(root, dst_dir)
                src_root = os.path.join(src_dir, rel_root) if rel_root != '.' else src_dir
                
                for file in files:
                    src_file = os.path.join(src_root, file)
                    dst_file = os.path.join(root, file)
                    
                    if not os.path.exists(src_file):
                        try:
                            os.remove(dst_file)
                            stats['deleted'] += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete {dst_file}: {e}")
                            stats['errors'] += 1
        
        return stats
    except Exception as e:
        raise FileIOException(f"Failed to sync directories: {e}", e)

# Private helper functions
def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy arrays and other objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

def _match_pattern(filename: str, pattern: str) -> bool:
    """Check if filename matches pattern (substring or glob)."""
    import fnmatch
    
    # Try glob pattern first
    if '*' in pattern or '?' in pattern or '[' in pattern:
        return fnmatch.fnmatch(filename, pattern)
    else:
        # Simple substring match
        return pattern in filename

def _get_mime_type(filepath: str) -> str:
    """Get MIME type of a file."""
    try:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(filepath)
        return mime_type or 'application/octet-stream'
    except Exception:
        return 'application/octet-stream'

def _detect_encoding(filepath: str) -> str:
    """Detect text file encoding."""
    try:
        import chardet
        with open(filepath, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
        result = chardet.detect(raw_data)
        return result.get('encoding', 'utf-8')
    except Exception:
        return 'utf-8'
