# src/metadata/__init__.py
"""
Metadata Extraction Module for SVELTE Framework.
Extracts, validates, and structures model metadata for comprehensive analysis.
"""

from .metadata_extractor import (
    MetadataExtractor,
    MetadataExtractionError,
    ModelArchitecture,
    LayerInfo,
    AttentionInfo,
    ParameterInfo,
    MetadataValidator,
    MetadataFormatter
)

__all__ = [
    'MetadataExtractor',
    'MetadataExtractionError',
    'ModelArchitecture', 
    'LayerInfo',
    'AttentionInfo',
    'ParameterInfo',
    'MetadataValidator',
    'MetadataFormatter'
]

__version__ = '1.0.0'