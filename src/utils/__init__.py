# src/utils/__init__.py
"""
Utilities Module for SVELTE Framework.
Provides file I/O operations, serialization, and utility functions.
"""

from .file_io import (
    read_binary_file,
    write_json,
    read_json,
    ensure_dir,
    copy_file,
    read_lines,
    write_lines,
    read_struct,
    safe_remove,
    list_files,
    file_size,
    is_json_file,
    move_file,
    FileIOException
)

__all__ = [
    'read_binary_file',
    'write_json',
    'read_json',
    'ensure_dir',
    'copy_file',
    'read_lines',
    'write_lines',
    'read_struct',
    'safe_remove',
    'list_files',
    'file_size',
    'is_json_file',
    'move_file',
    'FileIOException'
]

__version__ = '1.0.0'