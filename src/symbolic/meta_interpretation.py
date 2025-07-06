# src/symbolic/meta_interpretation.py
"""
Meta-Interpretation Synthesis Module for SVELTE Framework.
Integrates findings across modules to produce cohesive model interpretations.
author: Morpheus
date: 2025-05-01
description: This module synthesizes findings from various modules to create a unified interpretation of the model's behavior.
version: 0.1.0
ID: 002
SHA-256: abcdef1234567890abcdef1234567890abcdef123456
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor
import json
import itertools
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import argparse

# Configure logging
logger = logging.getLogger(__name__)

class InterpretationLevel(Enum):
 """Hierarchical levels of model interpretation."""
 PRIMITIVE = 0  # Raw patterns, activations, weights
 FUNCTIONAL = 1  # Functional units like attention heads, neurons
 BEHAVIORAL = 2  # Observable behaviors like token prediction
 SEMANTIC = 3    # Meaning and concepts
 EMERGENT = 4    # High-level capabilities and emergent properties


class ConflictResolutionStrategy(Enum):
 """Strategies for resolving conflicts between interpretations."""
 CONFIDENCE_WEIGHTED = "confidence_weighted"
 EVIDENCE_BASED = "evidence_based"
 CONSENSUS = "consensus"
 HIERARCHICAL = "hierarchical"
 CONTEXT_DEPENDENT = "context_dependent"


@dataclass
class InterpretationNode:
 """Representation of a single interpretation finding."""
 id: str
 source_module: str
 description: str
 level: InterpretationLevel
 confidence: float
 evidence: List[Dict[str, Any]]
 dependencies: Set[str] = None
 conflicts: Set[str] = None
 metadata: Dict[str, Any] = None
 
 def __post_init__(self):
  if self.dependencies is None:
   self.dependencies = set()
  if self.conflicts is None:
   self.conflicts = set()
  if self.metadata is None:
   self.metadata = {}


@dataclass
class TaxonomicClass:
 """Classification of model components in the taxonomy."""
 name: str
 description: str
 properties: Dict[str, Any]
 instances: List[str]
 parent: Optional[str] = None
 children: List[str] = None
 exemplars: List[str] = None
 similarity_threshold: float = 0.75
 
 def __post_init__(self):
  if self.children is None:
   self.children = []
  if self.exemplars is None:
   self.exemplars = []


class InterpretationGraph:
 """Graph structure for organizing interpretations and their relationships."""
 
 def __init__(self):
  self.graph = nx.DiGraph()
  self.node_index = {}  # Maps node IDs to their objects
  
 def add_node(self, node: InterpretationNode):
  """Add an interpretation node to the graph."""
  self.graph.add_node(node.id, data=node)
  self.node_index[node.id] = node
  
 def add_edge(self, source_id: str, target_id: str, edge_type: str, weight: float = 1.0):
  """Add a directed edge between interpretation nodes."""
  if source_id not in self.node_index or target_id not in self.node_index:
   raise ValueError(f"Nodes must exist in graph: {source_id} -> {target_id}")
  
  self.graph.add_edge(source_id, target_id, type=edge_type, weight=weight)
  
 def get_subgraph_by_level(self, level: InterpretationLevel) -> nx.DiGraph:
  """Extract subgraph containing only nodes at the specified level."""
  nodes = [node_id for node_id, node in self.node_index.items() 
    if node.level == level]
  return self.graph.subgraph(nodes)
 
 def find_conflicts(self) -> List[Tuple[str, str, float]]:
  """Find pairs of nodes with conflicting interpretations."""
  conflicts = []
  for node_id, node in self.node_index.items():
   for conflict_id in node.conflicts:
    if conflict_id in self.node_index:
     conflicts.append((node_id, conflict_id, 
         self._calculate_conflict_severity(node, self.node_index[conflict_id])))
  return conflicts
 
 def _calculate_conflict_severity(self, node1: InterpretationNode, node2: InterpretationNode) -> float:
  """Calculate the severity of conflict between two nodes."""
  # Higher confidence interpretations have more severe conflicts
  return (node1.confidence + node2.confidence) / 2.0


class MetaInterpretationSynthesisModule:
 """
 Integrates findings across modules to produce cohesive model interpretations.
 
 This module synthesizes outputs from various analysis modules within the SVELTE 
 framework to create a unified understanding of model behavior, organized in a 
 multi-level abstraction hierarchy with confidence-scored interpretations.
 """
 
 def __init__(self, module_outputs: Dict[str, Any], 
     conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.EVIDENCE_BASED,
     confidence_threshold: float = 0.6,
     max_workers: int = 4):
  """
  Initialize the Meta-Interpretation Synthesis Module.
  
  Args:
   module_outputs: Dictionary mapping module names to their output data structures
   conflict_strategy: Strategy for resolving conflicting interpretations
   confidence_threshold: Minimum confidence threshold for including interpretations
   max_workers: Maximum number of worker threads for parallel processing
  """
  self.module_outputs = module_outputs
  self.conflict_strategy = conflict_strategy
  self.confidence_threshold = confidence_threshold
  self.max_workers = max_workers
  
  # Core data structures
  self.interpretation_graph = InterpretationGraph()
  self.abstraction_hierarchy = {}  # Maps levels to subgraphs
  self.taxonomy = {}  # Taxonomic classification system
  self.integrated_interpretation = {}  # Final synthesized output
  
  # Module-specific processors
  self.processors = {
   "attention_topology": self._process_attention_topology,
   "memory_pattern": self._process_memory_patterns,
   "graph_builder": self._process_architecture_graph,
   # Add other module processors here
  }
  
  # Configure logging
  self.logger = logging.getLogger(__name__)
  self.logger.setLevel(logging.INFO)
 
 def synthesize(self) -> Dict[str, Any]:
  """
  Synthesize findings from all modules into a cohesive interpretation.
  
  This method performs the core integration work:
  1. Extract and normalize interpretations from each module
  2. Identify relationships and dependencies between interpretations
  3. Detect and resolve conflicts using the selected strategy
  4. Assign confidence scores based on evidence and coherence
  5. Organize into a structured representation of model behavior
  
  Returns:
   Dict containing the synthesized interpretation of the model
  """
  self.logger.info("Beginning meta-interpretation synthesis process")
  
  # Extract interpretations from each module
  self._extract_module_interpretations()
  
  # Build relationships between interpretations
  self._build_interpretation_relationships()
  
  # Detect and resolve conflicts
  conflicts = self._detect_and_resolve_conflicts()
  
  # Build the abstraction hierarchy
  self.build_abstraction_hierarchy()
  
  # Generate taxonomic classification
  self.generate_taxonomy()
  
  # Integrate everything into final interpretation
  self._integrate_findings()
  
  self.logger.info(f"Synthesis complete: {len(self.interpretation_graph.node_index)} interpretations, "
        f"{len(conflicts)} conflicts resolved")
  
  return self.integrated_interpretation
 
 def _extract_module_interpretations(self):
  """Extract interpretation nodes from each module's output."""
  for module_name, output in self.module_outputs.items():
   self.logger.debug(f"Processing output from module: {module_name}")
   
   if module_name in self.processors:
    # Use specialized processor for known modules
    interpretations = self.processors[module_name](output)
   else:
    # Generic processor for unknown modules
    interpretations = self._generic_interpretation_processor(module_name, output)
   
   # Add extracted interpretations to the graph
   for interp in interpretations:
    if interp.confidence >= self.confidence_threshold:
     self.interpretation_graph.add_node(interp)
 
 def _process_attention_topology(self, output: Dict[str, Any]) -> List[InterpretationNode]:
  """Process outputs from the attention topology module."""
  interpretations = []
  
  # Extract geometric patterns from attention manifolds
  if 'topology_metrics' in output:
   metrics = output['topology_metrics']
   
   # Analyze curvature patterns
   if hasattr(metrics, 'curvature') and metrics.curvature is not None:
    # Find regions of high curvature (information bottlenecks)
    high_curvature_regions = np.where(metrics.curvature > np.percentile(metrics.curvature, 90))
    
    for i, region in enumerate(high_curvature_regions[0]):
     interpretations.append(InterpretationNode(
      id=f"att_curv_{i}",
      source_module="attention_topology",
      description=f"Information bottleneck in attention at region {region}",
      level=InterpretationLevel.FUNCTIONAL,
      confidence=0.7 + 0.2 * (metrics.curvature[region] / np.max(metrics.curvature)),
      evidence=[{"type": "curvature", "value": float(metrics.curvature[region])}]
     ))
   
   # Analyze eigenvalues for principal directions
   if hasattr(metrics, 'eigenvalues') and metrics.eigenvalues is not None:
    # Significant eigenvalues suggest important attention dimensions
    sig_eigenvalues = np.where(metrics.eigenvalues > np.percentile(metrics.eigenvalues, 80))[0]
    
    for i, idx in enumerate(sig_eigenvalues):
     interpretations.append(InterpretationNode(
      id=f"att_eigen_{i}",
      source_module="attention_topology",
      description=f"Principal attention direction along dimension {idx}",
      level=InterpretationLevel.FUNCTIONAL,
      confidence=0.65 + 0.3 * (metrics.eigenvalues[idx] / np.max(metrics.eigenvalues)),
      evidence=[{"type": "eigenvalue", "value": float(metrics.eigenvalues[idx])}]
     ))
  
  return interpretations
 
 def _process_memory_patterns(self, output: Dict[str, Any]) -> List[InterpretationNode]:
  """Process outputs from the memory pattern recognition module."""
  interpretations = []
  
  # Extract memory motifs
  if 'memory_motifs' in output:
   motifs = output['memory_motifs']
   
   for i, motif in enumerate(motifs):
    # Convert PatternType to InterpretationLevel
    level_map = {
     "recurrent": InterpretationLevel.FUNCTIONAL,
     "attention": InterpretationLevel.FUNCTIONAL,
     "gating": InterpretationLevel.FUNCTIONAL,
     "residual": InterpretationLevel.PRIMITIVE,
     "feedforward": InterpretationLevel.PRIMITIVE,
     "cross_layer": InterpretationLevel.BEHAVIORAL
    }
    
    level = level_map.get(motif.pattern_type.value, InterpretationLevel.FUNCTIONAL)
    
    interpretations.append(InterpretationNode(
     id=f"mem_motif_{i}",
     source_module="memory_pattern",
     description=f"{motif.pattern_type.value.capitalize()} memory pattern: {motif.name}",
     level=level,
     confidence=min(0.9, motif.strength),
     evidence=[{
      "type": "memory_motif", 
      "strength": motif.strength,
      "frequency": motif.frequency,
      "variance": motif.variance
     }],
     metadata={
      "layer_ids": motif.layer_ids,
      "connections": list(motif.connections) if motif.connections else []
     }
    ))
  
  return interpretations
 
 def _process_architecture_graph(self, output: Dict[str, Any]) -> List[InterpretationNode]:
  """Process outputs from the architecture graph builder module."""
  interpretations = []
  
  # Process graph structure if available
  if 'graph' in output and isinstance(output['graph'], nx.Graph):
   graph = output['graph']
   
   # Identify highly connected nodes (hubs)
   centrality = nx.degree_centrality(graph)
   hubs = {node: cent for node, cent in centrality.items() 
       if cent > np.percentile(list(centrality.values()), 90)}
   
   for i, (node, centrality) in enumerate(hubs.items()):
    interpretations.append(InterpretationNode(
     id=f"arch_hub_{i}",
     source_module="graph_builder",
     description=f"Information hub at layer {node}",
     level=InterpretationLevel.FUNCTIONAL,
     confidence=0.7 + 0.25 * centrality,
     evidence=[{"type": "centrality", "value": centrality}],
     metadata={"node_id": node}
    ))
   
   # Identify strongly connected components (functional blocks)
   components = list(nx.strongly_connected_components(graph)) if nx.is_directed(graph) \
       else list(nx.connected_components(graph))
   
   for i, component in enumerate(components):
    if len(component) > 3:  # Only consider non-trivial components
     interpretations.append(InterpretationNode(
      id=f"arch_component_{i}",
      source_module="graph_builder",
      description=f"Functional block with {len(component)} layers",
      level=InterpretationLevel.FUNCTIONAL,
      confidence=0.75,
      evidence=[{"type": "component_size", "value": len(component)}],
      metadata={"component_nodes": list(component)}
     ))
   
   # Identify bottlenecks
   if nx.is_directed(graph):
    cuts = nx.minimum_edge_cut(graph.to_undirected())
    if cuts:
     interpretations.append(InterpretationNode(
      id="arch_bottleneck",
      source_module="graph_builder",
      description=f"Information bottleneck with {len(cuts)} critical connections",
      level=InterpretationLevel.BEHAVIORAL,
      confidence=0.8,
      evidence=[{"type": "edge_cut", "value": len(cuts)}],
      metadata={"bottleneck_edges": list(cuts)}
     ))
  
  return interpretations
 
 def _generic_interpretation_processor(self, module_name: str, output: Dict[str, Any]) -> List[InterpretationNode]:
  """Generic processor for modules without specialized processors."""
  interpretations = []
  
  # Try to extract interpretations from standard fields
  if isinstance(output, dict):
   # Look for findings, interpretations, or results keys
   for key in ['findings', 'interpretations', 'results', 'outputs']:
    if key in output and isinstance(output[key], list):
     for i, item in enumerate(output[key]):
      if isinstance(item, dict):
       # Try to extract standard fields
       desc = item.get('description', f"Finding from {module_name}")
       conf = item.get('confidence', 0.5)
       
       interpretations.append(InterpretationNode(
        id=f"{module_name}_{key}_{i}",
        source_module=module_name,
        description=desc,
        level=InterpretationLevel.FUNCTIONAL,  # Default level
        confidence=conf,
        evidence=[{"type": "generic", "source": f"{module_name}.{key}"}],
        metadata=item
       ))
  
  self.logger.debug(f"Generic processor extracted {len(interpretations)} interpretations from {module_name}")
  return interpretations
 
 def _build_interpretation_relationships(self):
  """Build relationships between interpretation nodes."""
  nodes = list(self.interpretation_graph.node_index.values())
  
  # Build dependencies based on layer references
  layer_to_nodes = defaultdict(list)
  for node in nodes:
   # Extract layer IDs from metadata
   layer_ids = []
   if 'metadata' in node.__dict__ and node.metadata:
    if 'layer_ids' in node.metadata:
     layer_ids.extend(node.metadata['layer_ids'])
    if 'component_nodes' in node.metadata:
     layer_ids.extend(node.metadata['component_nodes'])
   
   # Map layers to their nodes
   for layer_id in layer_ids:
    layer_to_nodes[layer_id].append(node.id)
  
  # Create edges between nodes sharing layers
  for layer_id, node_ids in layer_to_nodes.items():
   for node1, node2 in itertools.combinations(node_ids, 2):
    # Create bidirectional relationship as they may inform each other
    self.interpretation_graph.add_edge(node1, node2, "layer_association", 0.5)
    self.interpretation_graph.add_edge(node2, node1, "layer_association", 0.5)
  
  # Link nodes at different abstraction levels
  for node1, node2 in itertools.combinations(nodes, 2):
   if node1.level.value + 1 == node2.level.value:
    # Check for potential causal relationship from lower to higher abstraction
    if self._check_potential_causal_relationship(node1, node2):
     self.interpretation_graph.add_edge(node1.id, node2.id, "abstraction", 0.7)
 
 def _check_potential_causal_relationship(self, lower_node: InterpretationNode, 
           higher_node: InterpretationNode) -> bool:
  """Check if there's a potential causal relationship between nodes at different levels."""
  # This is a simplified version - in production would use more sophisticated analysis
  # Look for overlapping terms in descriptions
  lower_terms = set(lower_node.description.lower().split())
  higher_terms = set(higher_node.description.lower().split())
  
  # Filter out common stopwords
  stopwords = {'a', 'the', 'in', 'at', 'of', 'and', 'with', 'for'}
  lower_terms = lower_terms - stopwords
  higher_terms = higher_terms - stopwords
  
  # If they share meaningful terms, there might be a relationship
  return len(lower_terms.intersection(higher_terms)) >= 2
 
 def _detect_and_resolve_conflicts(self) -> List[Tuple[str, str, str]]:
  """Detect and resolve conflicts between interpretations."""
  conflicts = self.interpretation_graph.find_conflicts()
  resolved_conflicts = []
  
  for node1_id, node2_id, severity in conflicts:
   node1 = self.interpretation_graph.node_index[node1_id]
   node2 = self.interpretation_graph.node_index[node2_id]
   
   # Apply the selected conflict resolution strategy
   resolution = self._apply_conflict_resolution(node1, node2)
   resolved_conflicts.append((node1_id, node2_id, resolution))
   
   self.logger.debug(f"Resolved conflict between {node1_id} and {node2_id}: {resolution}")
  
  return resolved_conflicts
 
 def _apply_conflict_resolution(self, node1: InterpretationNode, node2: InterpretationNode) -> str:
  """Apply the selected conflict resolution strategy."""
  if self.conflict_strategy == ConflictResolutionStrategy.CONFIDENCE_WEIGHTED:
   # Keep the interpretation with higher confidence
   if node1.confidence > node2.confidence:
    return f"Selected {node1.id} (higher confidence)"
   else:
    return f"Selected {node2.id} (higher confidence)"
    
  elif self.conflict_strategy == ConflictResolutionStrategy.EVIDENCE_BASED:
   # Evaluate evidence quality and quantity
   evidence1 = len(node1.evidence)
   evidence2 = len(node2.evidence)
   
   # Simple evidence counting (would be more sophisticated in production)
   if evidence1 > evidence2:
    return f"Selected {node1.id} (stronger evidence)"
   else:
    return f"Selected {node2.id} (stronger evidence)"
    
  elif self.conflict_strategy == ConflictResolutionStrategy.CONSENSUS:
   # Look for consensus among other nodes
   # This is simplified - would require more complex graph analysis
   return "Applied consensus resolution (simplified)"
   
  else:
   # Default strategy - higher abstraction level wins
   if node1.level.value > node2.level.value:
    return f"Selected {node1.id} (higher abstraction)"
   else:
    return f"Selected {node2.id} (higher abstraction)"
 
 def build_abstraction_hierarchy(self) -> Dict[InterpretationLevel, nx.DiGraph]:
  """
  Construct a multi-level abstraction hierarchy mapping concepts across levels.
  
  This method:
  1. Organizes interpretations into hierarchical levels
  2. Identifies relationships between concepts at different levels
  3. Creates mappings from low-level patterns to high-level behaviors
  4. Builds a conceptual framework for understanding model components
  
  Returns:
   Dictionary mapping InterpretationLevel to subgraphs at that level
  """
  self.logger.info("Building abstraction hierarchy")
  
  # Create subgraphs for each level
  for level in InterpretationLevel:
   self.abstraction_hierarchy[level] = self.interpretation_graph.get_subgraph_by_level(level)
   
   self.logger.debug(f"Level {level.name}: {len(self.abstraction_hierarchy[level])} nodes")
  
  # Create inter-level connections
  for lower_level, higher_level in zip(list(InterpretationLevel)[:-1], list(InterpretationLevel)[1:]):
   lower_nodes = [node for node in self.interpretation_graph.node_index.values() 
        if node.level == lower_level]
   higher_nodes = [node for node in self.interpretation_graph.node_index.values() 
         if node.level == higher_level]
   
   # For each higher-level concept, try to find supporting lower-level concepts
   for higher_node in higher_nodes:
    supporting_nodes = self._find_supporting_nodes(higher_node, lower_nodes)
    
    # Create connections in the graph
    for lower_node in supporting_nodes:
     self.interpretation_graph.add_edge(
      lower_node.id, 
      higher_node.id, 
      "supports", 
      weight=self._calculate_support_strength(lower_node, higher_node)
     )
  
  # Identify emergent properties (nodes at EMERGENT level with multiple supporters)
  emergent_nodes = [node for node in self.interpretation_graph.node_index.values() 
       if node.level == InterpretationLevel.EMERGENT]
  
  for node in emergent_nodes:
   predecessors = list(self.interpretation_graph.graph.predecessors(node.id))
   if len(predecessors) >= 3:
    # This is a significant emergent property with multiple lower-level supports
    node.metadata["is_significant_emergent"] = True
    node.confidence = min(0.95, node.confidence + 0.1)
  
  return self.abstraction_hierarchy
 
 def _find_supporting_nodes(self, higher_node: InterpretationNode, 
         lower_nodes: List[InterpretationNode]) -> List[InterpretationNode]:
  """Find lower-level nodes that potentially support a higher-level node."""
  supporting_nodes = []
  
  # This is a simplified approach - would use more sophisticated matching in production
  higher_terms = set(higher_node.description.lower().split())
  
  for lower_node in lower_nodes:
   lower_terms = set(lower_node.description.lower().split())
   
   # Calculate Jaccard similarity
   if higher_terms and lower_terms:
    similarity = len(higher_terms.intersection(lower_terms)) / len(higher_terms.union(lower_terms))
    
    if similarity > 0.15:  # Arbitrary threshold
     supporting_nodes.append(lower_node)
  
  return supporting_nodes
 
 def _calculate_support_strength(self, lower_node: InterpretationNode, 
          higher_node: InterpretationNode) -> float:
  """Calculate the strength of support from lower to higher node."""
  # Combine confidence scores and semantic relevance
  base_strength = (lower_node.confidence + higher_node.confidence) / 2.0
  
  # Adjust based on evidence overlap
  lower_evidence_types = {e["type"] for e in lower_node.evidence}
  higher_evidence_types = {e["type"] for e in higher_node.evidence}
  
  evidence_overlap = len(lower_evidence_types.intersection(higher_evidence_types))
  evidence_factor = 1.0 + (0.1 * evidence_overlap)
  
  return min(1.0, base_strength * evidence_factor)
 
 def generate_taxonomy(self) -> Dict[str, TaxonomicClass]:
  """
  Generate a functional taxonomy of model components and behaviors.
  
  This method:
  1. Classifies components based on functional characteristics
  2. Groups similar patterns across different parts of the model
  3. Creates a standardized vocabulary for model components
  4. Enables cross-model comparison of similar functional units
  
  Returns:
   Dictionary mapping class names to TaxonomicClass objects
  """
  self.logger.info("Generating functional taxonomy")
  
  # Step 1: Collect functional units from interpretations
  functional_units = [node for node in self.interpretation_graph.node_index.values() 
        if node.level == InterpretationLevel.FUNCTIONAL]
  
  # Step 2: Extract features for clustering
  if not functional_units:
   self.logger.warning("No functional units found for taxonomy generation")
   return {}
  
  # Step 3: Create feature vectors from node metadata and descriptions
  # This is simplified - would use embedding models in production
  feature_vectors = []
  for node in functional_units:
   # Create simple feature vector from confidence and evidence count
   features = [
    node.confidence,
    len(node.evidence),
    len(node.dependencies),
    len(node.conflicts)
   ]
   feature_vectors.append(features)
  
  feature_matrix = np.array(feature_vectors)
  
  # Step 4: Perform hierarchical clustering
  if len(feature_matrix) > 1:
   # Find optimal number of clusters using silhouette score
   best_score = -1
   best_n_clusters = 2  # Default
   
   for n_clusters in range(2, min(10, len(feature_matrix))):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(feature_matrix)
    
    if len(set(labels)) > 1:  # Ensure we have at least 2 clusters
     score = silhouette_score(feature_matrix, labels)
     if score > best_score:
      best_score = score
      best_n_clusters = n_clusters
   
   # Apply clustering with optimal number of clusters
   clustering = AgglomerativeClustering(n_clusters=best_n_clusters)
   labels = clustering.fit_predict(feature_matrix)
   
   # Create taxonomy classes from clusters
   for cluster_id in range(best_n_clusters):
    # Get nodes in this cluster
    cluster_nodes = [node for i, node in enumerate(functional_units) if labels[i] == cluster_id]
    
    if not cluster_nodes:
     continue
    
    # Find common themes in descriptions
    descriptions = [node.description for node in cluster_nodes]
    common_terms = self._extract_common_terms(descriptions)
    
    # Create class name from common terms
    class_name = "_".join(list(common_terms)[:2]) if common_terms else f"class_{cluster_id}"
    
    # Choose exemplar as node with highest confidence
    exemplar = max(cluster_nodes, key=lambda n: n.confidence)
    
    # Create taxonomic class
    self.taxonomy[class_name] = TaxonomicClass(
     name=class_name,
     description=f"Functional unit type: {' '.join(list(common_terms)[:5])}",
     properties={
      "avg_confidence": np.mean([n.confidence for n in cluster_nodes]),
      "count": len(cluster_nodes),
     },
     instances=[node.id for node in cluster_nodes],
     exemplars=[exemplar.id],
    )
  
  # Step 5: Create hierarchical relationships between classes
  self._build_taxonomy_hierarchy()
  
  self.logger.info(f"Generated taxonomy with {len(self.taxonomy)} classes")
  return self.taxonomy
 
 def _extract_common_terms(self, texts: List[str]) -> Set[str]:
  """Extract common terms from a list of texts."""
  if not texts:
   return set()
  
  # Tokenize and count term frequencies
  stopwords = {'a', 'the', 'in', 'at', 'of', 'and', 'with', 'for'}
  term_counts = defaultdict(int)
  
  for text in texts:
   terms = [t.lower() for t in text.split() if t.lower() not in stopwords and len(t) > 3]
   for term in set(terms):  # Count each term once per text
    term_counts[term] += 1
  
  # Find terms that appear in at least half of the texts
  common_threshold = max(1, len(texts) // 2)
  common_terms = {term for term, count in term_counts.items() if count >= common_threshold}
  
  return common_terms
 
 def _build_taxonomy_hierarchy(self):
  """Build hierarchical relationships between taxonomic classes."""
  classes = list(self.taxonomy.values())
  
  # Create parent-child relationships based on instance overlap
  for parent, child in itertools.permutations(classes, 2):
   parent_instances = set(parent.instances)
   child_instances = set(child.instances)
   
   # If significant overlap but not identical
   overlap = len(child_instances.intersection(parent_instances))
   if overlap > 0 and overlap < len(child_instances) and \
      overlap / len(child_instances) > 0.7:
    # child could be a specialization of parent
    child.parent = parent.name
    parent.children.append(child.name)
 
 def _integrate_findings(self):
  """Integrate all findings into the final interpretation structure."""
  # Create the integrated interpretation structure
  self.integrated_interpretation = {
   "summary": self._generate_summary(),
   "abstraction_levels": {level.name: self._summarize_level(level)
        for level in InterpretationLevel},
   "taxonomy": {name: self._serialize_taxonomic_class(tax_class)
        for name, tax_class in self.taxonomy.items()},
   "key_findings": self._extract_key_findings(),
   "conflicts": self._summarize_conflicts(),
   "confidence": self._calculate_overall_confidence()
  }
 
 def _generate_summary(self) -> Dict[str, Any]:
  """Generate an overall summary of findings."""
  # Count interpretations by level
  level_counts = {level.name: 0 for level in InterpretationLevel}
  for node in self.interpretation_graph.node_index.values():
   level_counts[node.level.name] += 1
  
  # Find most confident interpretation at EMERGENT level
  emergent_nodes = [node for node in self.interpretation_graph.node_index.values() 
       if node.level == InterpretationLevel.EMERGENT]
  top_emergent = max(emergent_nodes, key=lambda n: n.confidence) if emergent_nodes else None
  
  return {
   "total_interpretations": len(self.interpretation_graph.node_index),
   "level_distribution": level_counts,
   "top_level_finding": top_emergent.description if top_emergent else None,
   "taxonomy_classes": len(self.taxonomy),
  }
 
 def _summarize_level(self, level: InterpretationLevel) -> Dict[str, Any]:
  """Summarize interpretations at a specific level."""
  nodes = [node for node in self.interpretation_graph.node_index.values() 
      if node.level == level]
  
  if not nodes:
   return {"count": 0}
  
  # Sort by confidence
  nodes.sort(key=lambda n: n.confidence, reverse=True)
  
  return {
   "count": len(nodes),
   "avg_confidence": np.mean([n.confidence for n in nodes]),
   "top_findings": [{"id": n.id, "description": n.description, 
        "confidence": n.confidence} for n in nodes[:5]]
  }
 
 def _serialize_taxonomic_class(self, tax_class: TaxonomicClass) -> Dict[str, Any]:
  """Serialize a taxonomic class for output."""
  return {
   "name": tax_class.name,
   "description": tax_class.description,
   "properties": tax_class.properties,
   "instance_count": len(tax_class.instances),
   "parent": tax_class.parent,
   "children": tax_class.children,
  }
 
 def _extract_key_findings(self) -> List[Dict[str, Any]]:
  """Extract key findings across all levels."""
  # Get top 2 findings from each level by confidence
  key_findings = []
  
  for level in InterpretationLevel:
   nodes = [node for node in self.interpretation_graph.node_index.values() 
       if node.level == level]
   
   # Sort by confidence and take top 2
   nodes.sort(key=lambda n: n.confidence, reverse=True)
   for node in nodes[:2]:
    key_findings.append({
     "id": node.id,
     "level": level.name,
     "description": node.description,
     "confidence": node.confidence,
     "source_module": node.source_module
    })
  
  return key_findings
 
 def _summarize_conflicts(self) -> Dict[str, Any]:
  """Summarize conflict resolution results."""
  conflicts = self.interpretation_graph.find_conflicts()
  
  return {
   "total_conflicts": len(conflicts),
   "resolution_strategy": self.conflict_strategy.value,
   "severe_conflicts": len([c for c in conflicts if c[2] > 0.7])
  }
 
 def _calculate_overall_confidence(self) -> float:
  """Calculate overall confidence in the interpretation."""
  # Average confidence of top findings weighted by level
  level_weights = {
   InterpretationLevel.PRIMITIVE: 0.6,
   InterpretationLevel.FUNCTIONAL: 0.7,
   InterpretationLevel.BEHAVIORAL: 0.8,
   InterpretationLevel.SEMANTIC: 0.9,
   InterpretationLevel.EMERGENT: 1.0
  }
  
  weighted_confs = []
  for node in self.interpretation_graph.node_index.values():
   weight = level_weights.get(node.level, 0.5)
   weighted_confs.append(node.confidence * weight)
  
  if not weighted_confs:
   return 0.0
  
  return sum(weighted_confs) / len(weighted_confs)
 
 def export_to_json(self, filepath: str):
  """Export the integrated interpretation to a JSON file."""
  if not self.integrated_interpretation:
   self.synthesize()
  
  # Convert to serializable format
  serializable = json.loads(json.dumps(self.integrated_interpretation, default=lambda o: o.__dict__))
  
  with open(filepath, 'w') as f:
   json.dump(serializable, f, indent=2)
  
  self.logger.info(f"Exported interpretation to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="SVELTE Meta-Interpretation Synthesis CLI")
    parser.add_argument('model', type=str, help='Path to GGUF model file')
    args = parser.parse_args()
    from src.tensor_analysis.gguf_parser import GGUFParser
    gguf = GGUFParser(args.model)
    gguf.parse()
    from src.tensor_analysis.tensor_field import TensorFieldConstructor
    tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
    tensor_field = tensor_field_constructor.construct()
    module_outputs = {'tensor_field': tensor_field}
    meta_interpret = MetaInterpretationSynthesisModule(module_outputs)
    meta_interpret.synthesize()
    meta_interpret.build_abstraction_hierarchy()
    meta_interpret.generate_taxonomy()
    import json
    print(json.dumps({'meta_interpretation': getattr(meta_interpret, 'integrated_interpretation', None)}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
