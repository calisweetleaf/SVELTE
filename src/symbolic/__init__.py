# src/symbolic/__init__.py
"""
Symbolic Processing Module for SVELTE Framework.
Transforms tensor patterns into symbolic representations and extracts computational grammars.
"""

from .symbolic_mapping import SymbolicMappingModule
from .meta_interpretation import (
    MetaInterpretationSynthesisModule,
    InterpretationLevel,
    ConflictResolutionStrategy,
    InterpretationNode,
    TaxonomicClass,
    InterpretationGraph
)

__all__ = [
    'SymbolicMappingModule',
    'MetaInterpretationSynthesisModule', 
    'InterpretationLevel',
    'ConflictResolutionStrategy',
    'InterpretationNode',
    'TaxonomicClass',
    'InterpretationGraph'
]

__version__ = '1.0.0'