# src/tensor_analysis/__init__.py
"""
Tensor Analysis Module for SVELTE Framework.
Provides comprehensive tensor excavation, entropy analysis, and quantization handling.
"""

from .gguf_parser import GGUFParser, GGUFParserError, GGUFConstants
from .tensor_field import (
    TensorField,
    TensorFieldConstructor,
    TensorMetadata,
    TensorRelationship,
    TensorIndex,
    TensorQuantizationType,
    TensorRelationshipDetector,
    QuantizationMapper
)
from .quantization import QuantizationReconstructor
from .entropy_analysis import EntropyAnalysisModule
from .activation_sim import ActivationSpaceSimulator, ActivationMetrics

__all__ = [
    'GGUFParser',
    'GGUFParserError', 
    'GGUFConstants',
    'TensorField',
    'TensorFieldConstructor',
    'TensorMetadata',
    'TensorRelationship',
    'TensorIndex',
    'TensorQuantizationType',
    'TensorRelationshipDetector',
    'QuantizationMapper',
    'QuantizationReconstructor',
    'EntropyAnalysisModule',
    'ActivationSpaceSimulator',
    'ActivationMetrics'
]

__version__ = '1.0.0'