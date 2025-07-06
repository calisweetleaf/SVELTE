# src/model_architecture/__init__.py
"""
Model Architecture Analysis Module for SVELTE Framework.

This module provides comprehensive analysis of neural network architectures
through differential geometry, topology, and graph theory approaches.
"""

from .attention_topology import AttentionTopologySystem, TopologyMetrics, CurvatureMethod
from .graph_builder import ArchitectureGraphBuilder
from .memory_pattern import MemoryPatternRecognitionSystem

__all__ = [
    'AttentionTopologySystem',
    'TopologyMetrics', 
    'CurvatureMethod',
    'ArchitectureGraphBuilder',
    'MemoryPatternRecognitionSystem'
]

__version__ = '1.0.0'
