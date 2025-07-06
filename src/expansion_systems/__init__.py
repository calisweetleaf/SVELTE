# src/expansion_systems/__init__.py
"""
Expansion Systems Module for SVELTE Framework.
Advanced analysis capabilities including cross-model comparison, 
counterfactual analysis, and regulatory compliance.
"""

from .cross_model_comparison import (
    CrossModelComparisonSystem,
    ModelAlignment,
    ComparisonMetrics,
    SimilarityMeasure,
    DifferenceAnalysis
)
from .counterfactual_analysis import (
    CounterfactualAnalysisSystem,
    InterventionType,
    CausalHypothesis,
    CounterfactualResult,
    WhatIfScenario
)
from .regulatory_compliance import (
    RegulatoryComplianceSystem,
    ComplianceFramework,
    AuditReport,
    SafetyEvidence,
    TransparencyReport,
    FairnessAssessment
)

__all__ = [
    'CrossModelComparisonSystem',
    'ModelAlignment',
    'ComparisonMetrics', 
    'SimilarityMeasure',
    'DifferenceAnalysis',
    'CounterfactualAnalysisSystem',
    'InterventionType',
    'CausalHypothesis',
    'CounterfactualResult',
    'WhatIfScenario',
    'RegulatoryComplianceSystem',
    'ComplianceFramework',
    'AuditReport',
    'SafetyEvidence',
    'TransparencyReport',
    'FairnessAssessment'
]

__version__ = '1.0.0'