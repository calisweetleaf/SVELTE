# src/expansion_systems/cross_model_comparison.py
"""
Cross-Model Comparison System for SVELTE Framework.
Enables comparative analysis of symbolic structures across different models.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import itertools
from collections import defaultdict
import argparse

try:
    from scipy import stats
    from scipy.spatial.distance import cosine, euclidean, hamming
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class SimilarityMeasure(Enum):
    """Types of similarity measures for model comparison."""
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    REPRESENTATIONAL = "representational"
    BEHAVIORAL = "behavioral"
    ARCHITECTURAL = "architectural"

class ComparisonDimension(Enum):
    """Dimensions along which models can be compared."""
    LAYER_TOPOLOGY = "layer_topology"
    ATTENTION_PATTERNS = "attention_patterns"
    MEMORY_STRUCTURES = "memory_structures"
    SYMBOLIC_PATTERNS = "symbolic_patterns"
    ENTROPY_DISTRIBUTIONS = "entropy_distributions"
    PARAMETER_EFFICIENCY = "parameter_efficiency"
    CAPABILITY_COVERAGE = "capability_coverage"

@dataclass
class ModelMetadata:
    """Metadata for a model in comparison."""
    model_id: str
    name: str
    architecture: str
    parameter_count: int
    layer_count: int
    attention_heads: int
    hidden_size: int
    context_length: int
    training_data: Optional[str] = None
    version: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComparisonMetrics:
    """Metrics for model comparison results."""
    structural_similarity: float
    functional_similarity: float
    representational_distance: float
    behavioral_alignment: float
    confidence_score: float
    comparison_dimensions: Dict[str, float] = field(default_factory=dict)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelAlignment:
    """Alignment between two models."""
    model_a: str
    model_b: str
    alignment_score: float
    aligned_components: Dict[str, Tuple[str, str]]
    unaligned_components: Dict[str, List[str]]
    transformation_matrix: Optional[np.ndarray] = None
    alignment_confidence: float = 0.0

@dataclass
class DifferenceAnalysis:
    """Analysis of differences between models."""
    unique_to_model_a: List[str]
    unique_to_model_b: List[str]
    shared_components: List[str]
    capability_differences: Dict[str, Any]
    architectural_differences: Dict[str, Any]
    performance_differences: Dict[str, float] = field(default_factory=dict)

class CrossModelComparisonSystem:
    """
    System for comparing multiple models across various dimensions.
    
    Provides comprehensive comparison capabilities including structural analysis,
    functional similarity measurement, and behavioral alignment assessment.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, max_workers: int = 4):
        """
        Initialize cross-model comparison system.
        
        Args:
            similarity_threshold: Threshold for considering components similar
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.similarity_threshold = similarity_threshold
        self.max_workers = max_workers
        self.models = {}
        self.comparison_cache = {}
        self.alignment_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"CrossModelComparisonSystem initialized with threshold {similarity_threshold}")
    
    def register_model(self, model_id: str, analysis_results: Dict[str, Any], 
                      metadata: Optional[ModelMetadata] = None) -> None:
        """
        Register a model for comparison.
        
        Args:
            model_id: Unique identifier for the model
            analysis_results: Complete SVELTE analysis results for the model
            metadata: Optional metadata about the model
        """
        if not metadata:
            metadata = self._extract_metadata_from_results(model_id, analysis_results)
        
        self.models[model_id] = {
            "metadata": metadata,
            "analysis": analysis_results,
            "processed_features": self._extract_comparison_features(analysis_results)
        }
        
        # Clear relevant caches
        self._clear_caches_for_model(model_id)
        
        logger.info(f"Registered model {model_id} for comparison")
    
    def compare_models(self, model_a: str, model_b: str, 
                      dimensions: List[ComparisonDimension] = None) -> ComparisonMetrics:
        """
        Compare two models across specified dimensions.
        
        Args:
            model_a: ID of first model
            model_b: ID of second model
            dimensions: List of comparison dimensions to evaluate
            
        Returns:
            Comprehensive comparison metrics
        """
        if model_a not in self.models or model_b not in self.models:
            raise ValueError("Both models must be registered before comparison")
        
        if dimensions is None:
            dimensions = list(ComparisonDimension)
        
        cache_key = f"{model_a}_{model_b}_{hash(tuple(dimensions))}"
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
        
        logger.info(f"Comparing models {model_a} and {model_b}")
        
        # Extract model data
        data_a = self.models[model_a]
        data_b = self.models[model_b]
        
        # Initialize metrics
        metrics = ComparisonMetrics(
            structural_similarity=0.0,
            functional_similarity=0.0,
            representational_distance=0.0,
            behavioral_alignment=0.0,
            confidence_score=0.0
        )
        
        # Compute similarities across dimensions
        dimension_scores = {}
        detailed_analysis = {}
        
        for dimension in dimensions:
            score, analysis = self._compare_dimension(data_a, data_b, dimension)
            dimension_scores[dimension.value] = score
            detailed_analysis[dimension.value] = analysis
        
        # Aggregate scores
        metrics.structural_similarity = self._compute_structural_similarity(
            dimension_scores, detailed_analysis
        )
        metrics.functional_similarity = self._compute_functional_similarity(
            dimension_scores, detailed_analysis
        )
        metrics.representational_distance = self._compute_representational_distance(
            dimension_scores, detailed_analysis
        )
        metrics.behavioral_alignment = self._compute_behavioral_alignment(
            dimension_scores, detailed_analysis
        )
        
        # Calculate overall confidence
        metrics.confidence_score = self._calculate_confidence_score(
            dimension_scores, data_a, data_b
        )
        
        metrics.comparison_dimensions = dimension_scores
        metrics.detailed_analysis = detailed_analysis
        
        # Cache results
        self.comparison_cache[cache_key] = metrics
        
        logger.info(f"Comparison complete: structural={metrics.structural_similarity:.3f}, "
                   f"functional={metrics.functional_similarity:.3f}")
        
        return metrics
    
    def align_models(self, model_a: str, model_b: str) -> ModelAlignment:
        """
        Find optimal alignment between two models.
        
        Args:
            model_a: ID of first model
            model_b: ID of second model
            
        Returns:
            Model alignment with component mappings
        """
        cache_key = f"align_{model_a}_{model_b}"
        if cache_key in self.alignment_cache:
            return self.alignment_cache[cache_key]
        
        logger.info(f"Aligning models {model_a} and {model_b}")
        
        data_a = self.models[model_a]
        data_b = self.models[model_b]
        
        # Extract components for alignment
        components_a = self._extract_alignable_components(data_a)
        components_b = self._extract_alignable_components(data_b)
        
        # Compute component similarities
        similarity_matrix = self._compute_component_similarities(components_a, components_b)
        
        # Find optimal alignment using Hungarian algorithm (simplified)
        aligned_pairs, alignment_score = self._find_optimal_alignment(
            similarity_matrix, components_a, components_b
        )
        
        # Identify unaligned components
        aligned_a = set(pair[0] for pair in aligned_pairs)
        aligned_b = set(pair[1] for pair in aligned_pairs)
        
        unaligned_a = [comp for comp in components_a.keys() if comp not in aligned_a]
        unaligned_b = [comp for comp in components_b.keys() if comp not in aligned_b]
        
        # Compute transformation matrix if applicable
        transformation_matrix = self._compute_transformation_matrix(
            aligned_pairs, components_a, components_b
        )
        
        alignment = ModelAlignment(
            model_a=model_a,
            model_b=model_b,
            alignment_score=alignment_score,
            aligned_components={f"{a}_to_{b}": (a, b) for a, b in aligned_pairs},
            unaligned_components={
                model_a: unaligned_a,
                model_b: unaligned_b
            },
            transformation_matrix=transformation_matrix,
            alignment_confidence=self._calculate_alignment_confidence(
                similarity_matrix, aligned_pairs
            )
        )
        
        self.alignment_cache[cache_key] = alignment
        return alignment
    
    def analyze_differences(self, model_a: str, model_b: str) -> DifferenceAnalysis:
        """
        Analyze differences between two models.
        
        Args:
            model_a: ID of first model
            model_b: ID of second model
            
        Returns:
            Detailed difference analysis
        """
        logger.info(f"Analyzing differences between {model_a} and {model_b}")
        
        data_a = self.models[model_a]
        data_b = self.models[model_b]
        
        # Extract features for comparison
        features_a = set(data_a["processed_features"].keys())
        features_b = set(data_b["processed_features"].keys())
        
        # Find unique and shared features
        unique_a = list(features_a - features_b)
        unique_b = list(features_b - features_a)
        shared = list(features_a & features_b)
        
        # Analyze capability differences
        capabilities_a = set(data_a["metadata"].capabilities)
        capabilities_b = set(data_b["metadata"].capabilities)
        
        capability_differences = {
            "unique_to_a": list(capabilities_a - capabilities_b),
            "unique_to_b": list(capabilities_b - capabilities_a),
            "shared": list(capabilities_a & capabilities_b)
        }
        
        # Analyze architectural differences
        arch_differences = self._analyze_architectural_differences(data_a, data_b)
        
        # Analyze performance differences if available
        performance_differences = self._analyze_performance_differences(data_a, data_b)
        
        return DifferenceAnalysis(
            unique_to_model_a=unique_a,
            unique_to_model_b=unique_b,
            shared_components=shared,
            capability_differences=capability_differences,
            architectural_differences=arch_differences,
            performance_differences=performance_differences
        )
    
    def compare_model_families(self, model_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Compare families or groups of models.
        
        Args:
            model_groups: Dictionary mapping group names to lists of model IDs
            
        Returns:
            Comprehensive family comparison analysis
        """
        logger.info(f"Comparing {len(model_groups)} model families")
        
        family_analysis = {}
        
        for family_name, model_list in model_groups.items():
            if len(model_list) < 2:
                logger.warning(f"Family {family_name} has fewer than 2 models, skipping")
                continue
            
            # Compute intra-family similarities
            intra_similarities = []
            for model_a, model_b in itertools.combinations(model_list, 2):
                if model_a in self.models and model_b in self.models:
                    metrics = self.compare_models(model_a, model_b)
                    intra_similarities.append(metrics.structural_similarity)
            
            # Compute family statistics
            family_analysis[family_name] = {
                "model_count": len(model_list),
                "avg_similarity": np.mean(intra_similarities) if intra_similarities else 0.0,
                "similarity_std": np.std(intra_similarities) if intra_similarities else 0.0,
                "cohesion_score": np.mean(intra_similarities) if intra_similarities else 0.0,
                "models": model_list
            }
        
        # Compute inter-family comparisons
        inter_family_similarities = {}
        family_names = list(model_groups.keys())
        
        for family_a, family_b in itertools.combinations(family_names, 2):
            similarities = []
            for model_a in model_groups[family_a]:
                for model_b in model_groups[family_b]:
                    if model_a in self.models and model_b in self.models:
                        metrics = self.compare_models(model_a, model_b)
                        similarities.append(metrics.structural_similarity)
            
            inter_family_similarities[f"{family_a}_vs_{family_b}"] = {
                "avg_similarity": np.mean(similarities) if similarities else 0.0,
                "similarity_std": np.std(similarities) if similarities else 0.0
            }
        
        return {
            "family_analysis": family_analysis,
            "inter_family_similarities": inter_family_similarities,
            "overall_statistics": self._compute_overall_family_statistics(family_analysis)
        }
    
    def create_similarity_matrix(self, model_list: List[str] = None) -> np.ndarray:
        """
        Create similarity matrix for registered models.
        
        Args:
            model_list: Optional list of specific models to include
            
        Returns:
            Similarity matrix as numpy array
        """
        if model_list is None:
            model_list = list(self.models.keys())
        
        n_models = len(model_list)
        similarity_matrix = np.zeros((n_models, n_models))
        
        for i, model_a in enumerate(model_list):
            for j, model_b in enumerate(model_list):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:
                    metrics = self.compare_models(model_a, model_b)
                    similarity = metrics.structural_similarity
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def cluster_models(self, model_list: List[str] = None, 
                      method: str = "hierarchical") -> Dict[str, Any]:
        """
        Cluster models based on similarity.
        
        Args:
            model_list: Optional list of specific models to cluster
            method: Clustering method ("hierarchical" or "kmeans")
            
        Returns:
            Clustering results with assignments and analysis
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for clustering functionality")
        
        if model_list is None:
            model_list = list(self.models.keys())
        
        # Create similarity matrix
        similarity_matrix = self.create_similarity_matrix(model_list)
        distance_matrix = 1 - similarity_matrix
        
        if method == "hierarchical":
            # Perform hierarchical clustering
            linkage_matrix = linkage(distance_matrix, method='ward')
            
            # Determine optimal number of clusters
            n_clusters = min(len(model_list) // 2, 5)
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[f"cluster_{label}"].append(model_list[i])
            
            return {
                "method": method,
                "clusters": dict(clusters),
                "linkage_matrix": linkage_matrix,
                "silhouette_score": self._calculate_silhouette_score(
                    distance_matrix, cluster_labels
                ) if SKLEARN_AVAILABLE else None
            }
        
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    
    def generate_comparison_report(self, model_a: str, model_b: str, 
                                 output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.
        
        Args:
            model_a: ID of first model
            model_b: ID of second model
            output_path: Optional path to save report
            
        Returns:
            Detailed comparison report
        """
        logger.info(f"Generating comparison report for {model_a} vs {model_b}")
        
        # Get all analysis components
        metrics = self.compare_models(model_a, model_b)
        alignment = self.align_models(model_a, model_b)
        differences = self.analyze_differences(model_a, model_b)
        
        # Compile report
        report = {
            "comparison_summary": {
                "model_a": model_a,
                "model_b": model_b,
                "overall_similarity": metrics.structural_similarity,
                "confidence": metrics.confidence_score,
                "timestamp": str(np.datetime64('now'))
            },
            "detailed_metrics": {
                "structural_similarity": metrics.structural_similarity,
                "functional_similarity": metrics.functional_similarity,
                "representational_distance": metrics.representational_distance,
                "behavioral_alignment": metrics.behavioral_alignment,
                "dimension_scores": metrics.comparison_dimensions
            },
            "model_alignment": {
                "alignment_score": alignment.alignment_score,
                "aligned_components_count": len(alignment.aligned_components),
                "unaligned_components": alignment.unaligned_components,
                "alignment_confidence": alignment.alignment_confidence
            },
            "difference_analysis": {
                "unique_features_a": len(differences.unique_to_model_a),
                "unique_features_b": len(differences.unique_to_model_b),
                "shared_features": len(differences.shared_components),
                "capability_differences": differences.capability_differences,
                "architectural_differences": differences.architectural_differences
            },
            "recommendations": self._generate_recommendations(metrics, alignment, differences)
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Comparison report saved to {output_path}")
        
        return report
    
    def _extract_metadata_from_results(self, model_id: str, 
                                     analysis_results: Dict[str, Any]) -> ModelMetadata:
        """Extract metadata from analysis results."""
        metadata_section = analysis_results.get("metadata", {})
        
        return ModelMetadata(
            model_id=model_id,
            name=metadata_section.get("model_name", model_id),
            architecture=metadata_section.get("architecture", "unknown"),
            parameter_count=metadata_section.get("parameter_count", 0),
            layer_count=metadata_section.get("layer_count", 0),
            attention_heads=metadata_section.get("attention_heads", 0),
            hidden_size=metadata_section.get("hidden_size", 0),
            context_length=metadata_section.get("context_length", 0)
        )
    
    def _extract_comparison_features(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features suitable for comparison."""
        features = {}
        
        # Extract entropy features
        if "entropy" in analysis_results:
            entropy_data = analysis_results["entropy"]
            features["entropy_stats"] = self._compute_entropy_statistics(entropy_data)
        
        # Extract attention features
        if "attention" in analysis_results:
            attention_data = analysis_results["attention"]
            features["attention_patterns"] = self._extract_attention_features(attention_data)
        
        # Extract symbolic features
        if "symbolic" in analysis_results:
            symbolic_data = analysis_results["symbolic"]
            features["symbolic_patterns"] = self._extract_symbolic_features(symbolic_data)
        
        # Extract architectural features
        if "graph" in analysis_results:
            graph_data = analysis_results["graph"]
            features["architectural_features"] = self._extract_architectural_features(graph_data)
        
        return features
    
    def _compare_dimension(self, data_a: Dict[str, Any], data_b: Dict[str, Any], 
                         dimension: ComparisonDimension) -> Tuple[float, Dict[str, Any]]:
        """Compare models along a specific dimension."""
        if dimension == ComparisonDimension.LAYER_TOPOLOGY:
            return self._compare_layer_topology(data_a, data_b)
        elif dimension == ComparisonDimension.ATTENTION_PATTERNS:
            return self._compare_attention_patterns(data_a, data_b)
        elif dimension == ComparisonDimension.MEMORY_STRUCTURES:
            return self._compare_memory_structures(data_a, data_b)
        elif dimension == ComparisonDimension.SYMBOLIC_PATTERNS:
            return self._compare_symbolic_patterns(data_a, data_b)
        elif dimension == ComparisonDimension.ENTROPY_DISTRIBUTIONS:
            return self._compare_entropy_distributions(data_a, data_b)
        else:
            logger.warning(f"Comparison dimension {dimension} not implemented")
            return 0.0, {}
    
    def _compare_layer_topology(self, data_a: Dict[str, Any], 
                              data_b: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Compare layer topology between models."""
        # Extract layer counts and structures
        meta_a = data_a["metadata"]
        meta_b = data_b["metadata"]
        
        layer_diff = abs(meta_a.layer_count - meta_b.layer_count)
        max_layers = max(meta_a.layer_count, meta_b.layer_count)
        
        topology_similarity = 1.0 - (layer_diff / max_layers) if max_layers > 0 else 1.0
        
        analysis = {
            "layer_count_a": meta_a.layer_count,
            "layer_count_b": meta_b.layer_count,
            "layer_difference": layer_diff,
            "topology_similarity": topology_similarity
        }
        
        return topology_similarity, analysis
    
    def _compare_attention_patterns(self, data_a: Dict[str, Any], 
                                  data_b: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Compare attention patterns between models."""
        features_a = data_a["processed_features"].get("attention_patterns", {})
        features_b = data_b["processed_features"].get("attention_patterns", {})
        
        if not features_a or not features_b:
            return 0.0, {"error": "Attention features not available"}
        
        # Simple similarity based on feature overlap
        similarity = len(set(features_a.keys()) & set(features_b.keys())) / \
                    max(len(features_a), len(features_b))
        
        analysis = {
            "patterns_a": len(features_a),
            "patterns_b": len(features_b),
            "shared_patterns": len(set(features_a.keys()) & set(features_b.keys())),
            "attention_similarity": similarity
        }
        
        return similarity, analysis
    
    def _compare_memory_structures(self, data_a: Dict[str, Any], 
                                 data_b: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Compare memory structures between models."""
        # Placeholder implementation
        return 0.5, {"memory_comparison": "not_implemented"}
    
    def _compare_symbolic_patterns(self, data_a: Dict[str, Any], 
                                 data_b: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Compare symbolic patterns between models."""
        features_a = data_a["processed_features"].get("symbolic_patterns", {})
        features_b = data_b["processed_features"].get("symbolic_patterns", {})
        
        if not features_a or not features_b:
            return 0.0, {"error": "Symbolic features not available"}
        
        # Compare pattern counts and types
        similarity = len(set(features_a.keys()) & set(features_b.keys())) / \
                    max(len(features_a), len(features_b))
        
        analysis = {
            "patterns_a": len(features_a),
            "patterns_b": len(features_b),
            "shared_patterns": len(set(features_a.keys()) & set(features_b.keys())),
            "symbolic_similarity": similarity
        }
        
        return similarity, analysis
    
    def _compare_entropy_distributions(self, data_a: Dict[str, Any], 
                                     data_b: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Compare entropy distributions between models."""
        features_a = data_a["processed_features"].get("entropy_stats", {})
        features_b = data_b["processed_features"].get("entropy_stats", {})
        
        if not features_a or not features_b:
            return 0.0, {"error": "Entropy features not available"}
        
        # Compare statistical measures
        similarity = 0.0
        comparisons = 0
        
        for stat in ["mean", "std", "skewness", "kurtosis"]:
            if stat in features_a and stat in features_b:
                val_a, val_b = features_a[stat], features_b[stat]
                if val_a != 0 or val_b != 0:
                    similarity += 1.0 - abs(val_a - val_b) / max(abs(val_a), abs(val_b))
                    comparisons += 1
        
        final_similarity = similarity / comparisons if comparisons > 0 else 0.0
        
        analysis = {
            "entropy_stats_a": features_a,
            "entropy_stats_b": features_b,
            "statistical_similarity": final_similarity
        }
        
        return final_similarity, analysis
    
    def _compute_entropy_statistics(self, entropy_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute statistical measures from entropy data."""
        if isinstance(entropy_data, dict):
            # Aggregate entropy values
            all_values = []
            for tensor_name, entropy_vals in entropy_data.items():
                if isinstance(entropy_vals, np.ndarray):
                    all_values.extend(entropy_vals.flatten())
                elif isinstance(entropy_vals, (int, float)):
                    all_values.append(entropy_vals)
            
            if all_values:
                arr = np.array(all_values)
                return {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "skewness": float(stats.skew(arr)) if SCIPY_AVAILABLE else 0.0,
                    "kurtosis": float(stats.kurtosis(arr)) if SCIPY_AVAILABLE else 0.0
                }
        
        return {}
    
    def _extract_attention_features(self, attention_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features from attention analysis."""
        features = {}
        
        try:
            # Extract basic attention statistics
            features["attention_heads"] = attention_data.get("num_heads", 0)
            features["attention_layers"] = attention_data.get("num_layers", 0)
            
            # Extract attention patterns if available
            if "attention_weights" in attention_data:
                weights = attention_data["attention_weights"]
                if isinstance(weights, np.ndarray):
                    features["attention_entropy"] = float(-np.sum(weights * np.log(weights + 1e-10)))
                    features["attention_sparsity"] = float(np.sum(weights < 0.1) / weights.size)
                    features["attention_concentration"] = float(np.max(weights) - np.min(weights))
            
            # Extract attention pattern statistics
            if "attention_patterns" in attention_data:
                patterns = attention_data["attention_patterns"]
                features["pattern_diversity"] = len(set(patterns.keys())) if patterns else 0
                features["dominant_pattern_strength"] = max(patterns.values()) if patterns else 0.0
            
            # Extract head specialization metrics
            if "head_specialization" in attention_data:
                spec = attention_data["head_specialization"]
                features["specialization_score"] = float(np.mean(list(spec.values()))) if spec else 0.0
                features["specialization_variance"] = float(np.var(list(spec.values()))) if spec else 0.0
            
        except Exception as e:
            logger.warning(f"Error extracting attention features: {e}")
            features["extraction_error"] = str(e)
        
        return features
    
    def _extract_symbolic_features(self, symbolic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features from symbolic analysis."""
        features = {}
        
        try:
            patterns = symbolic_data.get("patterns", {})
            features["pattern_count"] = len(patterns)
            
            # Extract pattern complexity metrics
            if patterns:
                pattern_lengths = [len(str(pattern)) for pattern in patterns.keys()]
                features["avg_pattern_length"] = float(np.mean(pattern_lengths))
                features["pattern_length_variance"] = float(np.var(pattern_lengths))
                
                # Extract pattern frequency statistics
                pattern_freqs = list(patterns.values())
                if pattern_freqs:
                    features["pattern_frequency_mean"] = float(np.mean(pattern_freqs))
                    features["pattern_frequency_std"] = float(np.std(pattern_freqs))
                    features["pattern_frequency_entropy"] = float(-np.sum(
                        [f * np.log(f + 1e-10) for f in pattern_freqs]
                    ))
            
            # Extract symbolic complexity measures
            if "symbolic_complexity" in symbolic_data:
                complexity = symbolic_data["symbolic_complexity"]
                features["symbolic_complexity_score"] = float(complexity.get("score", 0.0))
                features["symbolic_depth"] = int(complexity.get("depth", 0))
                features["symbolic_breadth"] = int(complexity.get("breadth", 0))
            
            # Extract abstraction levels
            if "abstraction_levels" in symbolic_data:
                levels = symbolic_data["abstraction_levels"]
                features["abstraction_count"] = len(levels)
                features["max_abstraction_level"] = max(levels) if levels else 0
                features["abstraction_distribution"] = float(np.std(levels)) if levels else 0.0
            
        except Exception as e:
            logger.warning(f"Error extracting symbolic features: {e}")
            features["extraction_error"] = str(e)
        
        return features
    
    def _extract_architectural_features(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features from architectural graph."""
        features = {}
        
        try:
            # Basic graph metrics
            features["node_count"] = graph_data.get("node_count", 0)
            features["edge_count"] = graph_data.get("edge_count", 0)
            features["graph_density"] = graph_data.get("density", 0.0)
            
            # Graph connectivity metrics
            if "connectivity" in graph_data:
                connectivity = graph_data["connectivity"]
                features["avg_degree"] = float(connectivity.get("avg_degree", 0.0))
                features["max_degree"] = int(connectivity.get("max_degree", 0))
                features["clustering_coefficient"] = float(connectivity.get("clustering_coefficient", 0.0))
            
            # Graph structure metrics
            if "structure" in graph_data:
                structure = graph_data["structure"]
                features["diameter"] = int(structure.get("diameter", 0))
                features["avg_path_length"] = float(structure.get("avg_path_length", 0.0))
                features["modularity"] = float(structure.get("modularity", 0.0))
            
            # Layer-specific features
            if "layers" in graph_data:
                layers = graph_data["layers"]
                features["layer_count"] = len(layers)
                layer_sizes = [layer.get("size", 0) for layer in layers]
                features["avg_layer_size"] = float(np.mean(layer_sizes)) if layer_sizes else 0.0
                features["layer_size_variance"] = float(np.var(layer_sizes)) if layer_sizes else 0.0
            
            # Compute architectural complexity
            features["architectural_complexity"] = self._compute_architectural_complexity(graph_data)
            
        except Exception as e:
            logger.warning(f"Error extracting architectural features: {e}")
            features["extraction_error"] = str(e)
        
        return features
    
    def _compute_architectural_complexity(self, graph_data: Dict[str, Any]) -> float:
        """Compute architectural complexity score."""
        try:
            node_count = graph_data.get("node_count", 0)
            edge_count = graph_data.get("edge_count", 0)
            
            if node_count == 0:
                return 0.0
            
            # Compute complexity based on graph structure
            density = edge_count / max(node_count * (node_count - 1) / 2, 1)
            connectivity = graph_data.get("connectivity", {})
            clustering = connectivity.get("clustering_coefficient", 0.0)
            
            # Combine metrics for complexity score
            complexity = (density * 0.4 + clustering * 0.3 + 
                         min(node_count / 100, 1.0) * 0.3)
            
            return float(complexity)
            
        except Exception as e:
            logger.warning(f"Error computing architectural complexity: {e}")
            return 0.0
    
    def _compute_structural_similarity(self, dimension_scores: Dict[str, float], 
                                     detailed_analysis: Dict[str, Any]) -> float:
        """Compute overall structural similarity with weighted dimensions."""
        # Define weights for structural dimensions
        structural_weights = {
            "layer_topology": 0.4,
            "architectural_features": 0.3,
            "parameter_efficiency": 0.2,
            "capability_coverage": 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for dim, weight in structural_weights.items():
            if dim in dimension_scores:
                weighted_score += dimension_scores[dim] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _compute_functional_similarity(self, dimension_scores: Dict[str, float], 
                                     detailed_analysis: Dict[str, Any]) -> float:
        """Compute functional similarity with enhanced metrics."""
        # Define weights for functional dimensions
        functional_weights = {
            "attention_patterns": 0.4,
            "memory_structures": 0.3,
            "symbolic_patterns": 0.3
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for dim, weight in functional_weights.items():
            if dim in dimension_scores:
                weighted_score += dimension_scores[dim] * weight
                total_weight += weight
        
        # Apply functional complexity adjustment
        if total_weight > 0:
            base_score = weighted_score / total_weight
            
            # Adjust based on functional complexity alignment
            complexity_alignment = self._compute_complexity_alignment(detailed_analysis)
            adjusted_score = base_score * (0.7 + 0.3 * complexity_alignment)
            
            return min(adjusted_score, 1.0)
        
        return 0.0
    
    def _compute_complexity_alignment(self, detailed_analysis: Dict[str, Any]) -> float:
        """Compute how well complexity levels align between models."""
        try:
            # Extract complexity indicators from analysis
            attention_analysis = detailed_analysis.get("attention_patterns", {})
            symbolic_analysis = detailed_analysis.get("symbolic_patterns", {})
            
            alignments = []
            
            # Check attention complexity alignment
            if "attention_similarity" in attention_analysis:
                alignments.append(attention_analysis["attention_similarity"])
            
            # Check symbolic complexity alignment
            if "symbolic_similarity" in symbolic_analysis:
                alignments.append(symbolic_analysis["symbolic_similarity"])
            
            return float(np.mean(alignments)) if alignments else 0.5
            
        except Exception as e:
            logger.warning(f"Error computing complexity alignment: {e}")
            return 0.5
    
    def _compute_representational_distance(self, dimension_scores: Dict[str, float], 
                                         detailed_analysis: Dict[str, Any]) -> float:
        """Compute representational distance with statistical measures."""
        # Define dimensions that contribute to representational distance
        repr_dimensions = {
            "entropy_distributions": 0.4,
            "symbolic_patterns": 0.3,
            "attention_patterns": 0.3
        }
        
        distance_components = []
        
        for dim, weight in repr_dimensions.items():
            if dim in dimension_scores:
                # Convert similarity to distance
                distance = 1.0 - dimension_scores[dim]
                distance_components.append(distance * weight)
        
        if not distance_components:
            return 1.0
        
        # Compute weighted distance
        base_distance = sum(distance_components) / sum(repr_dimensions.values())
        
        # Apply statistical adjustments
        statistical_adjustment = self._compute_statistical_distance_adjustment(detailed_analysis)
        adjusted_distance = base_distance * (0.8 + 0.2 * statistical_adjustment)
        
        return min(max(adjusted_distance, 0.0), 1.0)
    
    def _compute_statistical_distance_adjustment(self, detailed_analysis: Dict[str, Any]) -> float:
        """Compute statistical adjustment for representational distance."""
        try:
            # Extract statistical measures
            entropy_analysis = detailed_analysis.get("entropy_distributions", {})
            
            if "statistical_similarity" in entropy_analysis:
                return 1.0 - entropy_analysis["statistical_similarity"]
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error computing statistical distance adjustment: {e}")
            return 0.5
    
    def _compute_behavioral_alignment(self, dimension_scores: Dict[str, float], 
                                    detailed_analysis: Dict[str, Any]) -> float:
        """Compute behavioral alignment with comprehensive metrics."""
        # Behavioral alignment considers multiple factors
        behavioral_factors = {
            "functional_similarity": 0.3,
            "attention_patterns": 0.25,
            "symbolic_patterns": 0.25,
            "architectural_compatibility": 0.2
        }
        
        alignment_score = 0.0
        total_weight = 0.0
        
        for factor, weight in behavioral_factors.items():
            if factor in dimension_scores:
                alignment_score += dimension_scores[factor] * weight
                total_weight += weight
        
        if total_weight == 0:
            # Fallback to average of available scores
            available_scores = [score for score in dimension_scores.values() if score > 0]
            return float(np.mean(available_scores)) if available_scores else 0.0
        
        base_alignment = alignment_score / total_weight
        
        # Apply behavioral consistency adjustment
        consistency_bonus = self._compute_behavioral_consistency(detailed_analysis)
        final_alignment = base_alignment * (0.9 + 0.1 * consistency_bonus)
        
        return min(final_alignment, 1.0)
    
    def _compute_behavioral_consistency(self, detailed_analysis: Dict[str, Any]) -> float:
        """Compute behavioral consistency score."""
        try:
            # Check consistency across different analysis dimensions
            consistency_indicators = []
            
            # Attention consistency
            attention_analysis = detailed_analysis.get("attention_patterns", {})
            if "patterns_a" in attention_analysis and "patterns_b" in attention_analysis:
                pattern_ratio = min(attention_analysis["patterns_a"], attention_analysis["patterns_b"]) / \
                              max(attention_analysis["patterns_a"], attention_analysis["patterns_b"])
                consistency_indicators.append(pattern_ratio)
            
            # Symbolic consistency
            symbolic_analysis = detailed_analysis.get("symbolic_patterns", {})
            if "patterns_a" in symbolic_analysis and "patterns_b" in symbolic_analysis:
                pattern_ratio = min(symbolic_analysis["patterns_a"], symbolic_analysis["patterns_b"]) / \
                              max(symbolic_analysis["patterns_a"], symbolic_analysis["patterns_b"])
                consistency_indicators.append(pattern_ratio)
            
            return float(np.mean(consistency_indicators)) if consistency_indicators else 0.5
            
        except Exception as e:
            logger.warning(f"Error computing behavioral consistency: {e}")
            return 0.5
    
    def _calculate_confidence_score(self, dimension_scores: Dict[str, float], 
                                  data_a: Dict[str, Any], data_b: Dict[str, Any]) -> float:
        """Calculate confidence in comparison results with robust metrics."""
        try:
            # Base confidence on coverage of comparison dimensions
            dimension_coverage = len(dimension_scores) / len(ComparisonDimension)
            
            # Data quality assessment
            features_a = data_a.get("processed_features", {})
            features_b = data_b.get("processed_features", {})
            
            # Check for extraction errors
            errors_a = sum(1 for f in features_a.values() if "extraction_error" in f)
            errors_b = sum(1 for f in features_b.values() if "extraction_error" in f)
            
            error_penalty = (errors_a + errors_b) / max(len(features_a) + len(features_b), 1)
            
            # Data completeness
            common_features = set(features_a.keys()) & set(features_b.keys())
            all_features = set(features_a.keys()) | set(features_b.keys())
            
            data_completeness = len(common_features) / max(len(all_features), 1)
            
            # Score consistency check
            score_variance = np.var(list(dimension_scores.values())) if dimension_scores else 0.0
            consistency_factor = 1.0 - min(score_variance, 0.5)  # Penalize high variance
            
            # Combine factors
            confidence = (dimension_coverage * 0.4 + 
                         data_completeness * 0.3 + 
                         consistency_factor * 0.2 + 
                         (1.0 - error_penalty) * 0.1)
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _clear_caches_for_model(self, model_id: str) -> None:
        """Clear caches involving specified model."""
        to_remove = []
        for key in self.comparison_cache:
            if model_id in key:
                to_remove.append(key)
        
        for key in to_remove:
            del self.comparison_cache[key]
        
        # Clear alignment cache
        to_remove = []
        for key in self.alignment_cache:
            if model_id in key:
                to_remove.append(key)
        
        for key in to_remove:
            del self.alignment_cache[key]
    
    def _extract_alignable_components(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract components that can be aligned between models."""
        components = {}
        
        # Extract layer components
        metadata = model_data["metadata"]
        for i in range(metadata.layer_count):
            components[f"layer_{i}"] = {"type": "layer", "index": i}
        
        # Extract attention components
        features = model_data["processed_features"]
        if "attention_patterns" in features:
            for pattern_name in features["attention_patterns"]:
                components[f"attention_{pattern_name}"] = {"type": "attention", "name": pattern_name}
        
        return components
    
    def _compute_component_similarities(self, components_a: Dict[str, Any], 
                                      components_b: Dict[str, Any]) -> np.ndarray:
        """Compute similarity matrix between components."""
        comp_list_a = list(components_a.keys())
        comp_list_b = list(components_b.keys())
        
        similarity_matrix = np.zeros((len(comp_list_a), len(comp_list_b)))
        
        for i, comp_a in enumerate(comp_list_a):
            for j, comp_b in enumerate(comp_list_b):
                similarity = self._compute_component_similarity(
                    components_a[comp_a], components_b[comp_b]
                )
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def _compute_component_similarity(self, comp_a: Dict[str, Any], 
                                    comp_b: Dict[str, Any]) -> float:
        """Compute enhanced similarity between two components."""
        try:
            # Type compatibility check
            if comp_a["type"] != comp_b["type"]:
                return 0.0
            
            # Base similarity for type match
            base_similarity = 0.6
            
            # Enhanced similarity based on component properties
            if comp_a["type"] == "layer":
                # Layer-specific similarity
                index_a = comp_a.get("index", 0)
                index_b = comp_b.get("index", 0)
                
                # Similar indices suggest similar roles
                index_similarity = 1.0 - abs(index_a - index_b) / max(index_a, index_b, 1)
                return base_similarity + 0.4 * index_similarity
                
            elif comp_a["type"] == "attention":
                # Attention-specific similarity
                name_a = comp_a.get("name", "")
                name_b = comp_b.get("name", "")
                
                # Simple string similarity
                if name_a and name_b:
                    name_similarity = len(set(name_a) & set(name_b)) / len(set(name_a) | set(name_b))
                    return base_similarity + 0.4 * name_similarity
            
            return base_similarity
            
        except Exception as e:
            logger.warning(f"Error computing component similarity: {e}")
            return 0.0
    
    def _find_optimal_alignment(self, similarity_matrix: np.ndarray, 
                              components_a: Dict[str, Any], 
                              components_b: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], float]:
        """Find optimal alignment using greedy approach."""
        comp_list_a = list(components_a.keys())
        comp_list_b = list(components_b.keys())
        
        aligned_pairs = []
        used_a = set()
        used_b = set()
        
        # Greedy alignment based on highest similarities
        while True:
            best_similarity = 0.0
            best_pair = None
            
            for i, comp_a in enumerate(comp_list_a):
                if comp_a in used_a:
                    continue
                for j, comp_b in enumerate(comp_list_b):
                    if comp_b in used_b:
                        continue
                    
                    if similarity_matrix[i, j] > best_similarity and \
                       similarity_matrix[i, j] >= self.similarity_threshold:
                        best_similarity = similarity_matrix[i, j]
                        best_pair = (comp_a, comp_b)
            
            if best_pair is None:
                break
            
            aligned_pairs.append(best_pair)
            used_a.add(best_pair[0])
            used_b.add(best_pair[1])
        
        # Calculate overall alignment score
        alignment_score = np.mean([similarity_matrix[comp_list_a.index(a), comp_list_b.index(b)] 
                                  for a, b in aligned_pairs]) if aligned_pairs else 0.0
        
        return aligned_pairs, alignment_score
    
    def _compute_transformation_matrix(self, aligned_pairs: List[Tuple[str, str]], 
                                     components_a: Dict[str, Any], 
                                     components_b: Dict[str, Any]) -> Optional[np.ndarray]:
        """Compute transformation matrix for aligned components."""
        # Placeholder - would compute actual transformation based on component features
        if len(aligned_pairs) < 2:
            return None
        
        # Return identity matrix as placeholder
        return np.eye(len(aligned_pairs))
    
    def _calculate_alignment_confidence(self, similarity_matrix: np.ndarray, 
                                      aligned_pairs: List[Tuple[str, str]]) -> float:
        """Calculate confidence in alignment results with improved metrics."""
        try:
            if not aligned_pairs or similarity_matrix.size == 0:
                return 0.0
            
            # Extract actual similarities for aligned pairs
            similarities = []
            for i, (comp_a, comp_b) in enumerate(aligned_pairs):
                # For production, would map component names to matrix indices
                # Using approximation based on alignment quality
                if i < min(similarity_matrix.shape):
                    similarities.append(similarity_matrix[i, i])
                else:
                    similarities.append(0.7)  # Conservative estimate
            
            # Base confidence on alignment quality
            avg_similarity = np.mean(similarities)
            similarity_consistency = 1.0 - np.std(similarities)
            
            # Alignment coverage
            total_possible = min(similarity_matrix.shape)
            alignment_coverage = len(aligned_pairs) / max(total_possible, 1)
            
            # Combined confidence
            confidence = (avg_similarity * 0.5 + 
                         similarity_consistency * 0.3 + 
                         alignment_coverage * 0.2)
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating alignment confidence: {e}")
            return 0.5
    
    def _analyze_performance_differences(self, data_a: Dict[str, Any], 
                                       data_b: Dict[str, Any]) -> Dict[str, float]:
        """Analyze performance differences between models with real metrics."""
        try:
            performance_diffs = {}
            
            # Model size efficiency
            meta_a = data_a["metadata"]
            meta_b = data_b["metadata"]
            
            if meta_a.parameter_count > 0 and meta_b.parameter_count > 0:
                performance_diffs["parameter_efficiency_ratio"] = \
                    meta_a.parameter_count / meta_b.parameter_count
            
            # Architectural efficiency
            if meta_a.layer_count > 0 and meta_b.layer_count > 0:
                performance_diffs["layer_efficiency_ratio"] = \
                    meta_a.layer_count / meta_b.layer_count
            
            # Memory efficiency estimation
            memory_a = meta_a.parameter_count * meta_a.hidden_size
            memory_b = meta_b.parameter_count * meta_b.hidden_size
            
            if memory_b > 0:
                performance_diffs["memory_efficiency_ratio"] = memory_a / memory_b
            
            # Attention efficiency
            if meta_a.attention_heads > 0 and meta_b.attention_heads > 0:
                performance_diffs["attention_efficiency_ratio"] = \
                    meta_a.attention_heads / meta_b.attention_heads
            
            # Complexity-based performance estimation
            features_a = data_a.get("processed_features", {})
            features_b = data_b.get("processed_features", {})
            
            complexity_a = self._estimate_computational_complexity(features_a)
            complexity_b = self._estimate_computational_complexity(features_b)
            
            if complexity_b > 0:
                performance_diffs["computational_complexity_ratio"] = complexity_a / complexity_b
            
            return performance_diffs
            
        except Exception as e:
            logger.warning(f"Error analyzing performance differences: {e}")
            return {"analysis_error": 1.0}
    
    def _estimate_computational_complexity(self, features: Dict[str, Any]) -> float:
        """Estimate computational complexity from features."""
        try:
            complexity = 0.0
            
            # Attention complexity
            attention_features = features.get("attention_patterns", {})
            if "attention_heads" in attention_features:
                complexity += attention_features["attention_heads"] * 0.1
            
            # Symbolic complexity
            symbolic_features = features.get("symbolic_patterns", {})
            if "pattern_count" in symbolic_features:
                complexity += symbolic_features["pattern_count"] * 0.05
            
            # Architectural complexity
            arch_features = features.get("architectural_features", {})
            if "node_count" in arch_features:
                complexity += arch_features["node_count"] * 0.01
            
            return max(complexity, 0.1)  # Minimum complexity
            
        except Exception as e:
            logger.warning(f"Error estimating computational complexity: {e}")
            return 1.0
    
    def _compute_overall_family_statistics(self, family_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Compute overall statistics across families."""
        all_cohesions = [family["cohesion_score"] for family in family_analysis.values()]
        all_similarities = [family["avg_similarity"] for family in family_analysis.values()]
        
        return {
            "mean_family_cohesion": np.mean(all_cohesions) if all_cohesions else 0.0,
            "std_family_cohesion": np.std(all_cohesions) if all_cohesions else 0.0,
            "mean_family_similarity": np.mean(all_similarities) if all_similarities else 0.0,
            "total_families": len(family_analysis)
        }
    
    def _calculate_silhouette_score(self, distance_matrix: np.ndarray, 
                                  cluster_labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering."""
        if SKLEARN_AVAILABLE:
            from sklearn.metrics import silhouette_score
            return silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        return 0.0
    
    def _generate_recommendations(self, metrics: ComparisonMetrics, 
                                alignment: ModelAlignment, 
                                differences: DifferenceAnalysis) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        if metrics.structural_similarity > 0.8:
            recommendations.append("Models show high structural similarity - consider architectural consolidation")
        elif metrics.structural_similarity < 0.3:
            recommendations.append("Models are structurally very different - investigate unique advantages")
        
        if alignment.alignment_score > 0.7:
            recommendations.append("Strong component alignment enables potential knowledge transfer")
        
        if len(differences.shared_components) / max(
            len(differences.unique_to_model_a) + len(differences.unique_to_model_b), 1
        ) > 0.5:
            recommendations.append("High component overlap suggests redundancy potential")
        
        return recommendations

def main():
    """CLI entry point for cross-model comparison."""
    parser = argparse.ArgumentParser(description="SVELTE Cross-Model Comparison CLI")
    parser.add_argument('--models', nargs='+', required=True, help='Model IDs to compare')
    parser.add_argument('--output', '-o', type=str, help='Output file for comparison report')
    parser.add_argument('--cluster', action='store_true', help='Perform clustering analysis')
    args = parser.parse_args()
    
    # This would normally load actual model analysis results
    print(f"Cross-model comparison for models: {args.models}")
    print("Note: This is a CLI stub. Actual implementation would load model analysis results.")
    
    comparison_system = CrossModelComparisonSystem()
    
    # Generate mock comparison
    if len(args.models) >= 2:
        print(f"Would compare {args.models[0]} vs {args.models[1]}")
        if args.output:
            print(f"Would save report to {args.output}")
    
    if args.cluster and len(args.models) > 2:
        print(f"Would perform clustering on {len(args.models)} models")

if __name__ == "__main__":
    main()