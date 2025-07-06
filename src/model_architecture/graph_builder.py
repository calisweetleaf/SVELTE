# src/model_architecture/graph_builder.py
"""
Architecture Graph Builder for SVELTE Framework.
Reconstructs computational graphs, layer connectivity, and skip/residual paths.
author: Morpheus
date: 2025-05-01
version: 1.0.0
description: This module builds a directed graph representation of the model architecture.
It captures the relationships between different layers, including skip connections and residual paths.
It is designed to work with the metadata extracted from the model, providing a structured way to analyze and visualize the architecture.
id: 002
SHA-256: 3a1b2c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z7a8b9c0d1e2f3g4h5i6j7k8l9m0n1o2p3q4r5s6t7u8v9w0x1y2z3a4b5c6d7e8f9g0h1i2j3k4l5m6n7o8p9q0r1s2t3u4v5w6x7y8z9a0b1c2d3e4f5g6h7i8j9k0l1m2n3o4p5q6r7s8t9u0v1w2x3y4z5a6b7c8d9e0f1g2h3i4j5k6l7m8n

"""
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Iterator, Generator
import logging
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict, Counter
import json
import hashlib
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import warnings
import argparse
import sys

# Configure logging
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConnectionType(Enum):
 """Enum for different types of layer connections in neural networks."""
 SEQUENTIAL = auto()
 RESIDUAL = auto()
 SKIP = auto()
 PARALLEL = auto()
 MERGE = auto()
 BRANCH = auto()
 RECURRENT = auto()

@dataclass
class LayerInfo:
 """Dataclass to store information about a layer in the neural network."""
 id: str
 name: str
 type: str
 params: int
 input_shape: Optional[List[int]] = None
 output_shape: Optional[List[int]] = None
 activation: Optional[str] = None
 trainable: bool = True
 
 def __post_init__(self):
  """Validate and process the layer information after initialization."""
  if not self.id or not self.name or not self.type:
   raise ValueError("Layer id, name, and type are required")
  
  # Normalize the layer type to a standard format
  self.type = self.type.lower().strip()
  
  # Generate a hash identifier for the layer
  self.hash = hashlib.md5(f"{self.id}:{self.name}:{self.type}".encode()).hexdigest()

class ArchitectureGraphBuilder:
 """
 Builds a directed graph representation of neural network architecture.
 
 This class processes model metadata to construct a NetworkX directed graph
 where nodes represent layers and edges represent connections between layers.
 It provides methods to analyze the architecture, identify different connection
 patterns, and visualize the model structure.
 """
 
 def __init__(self, metadata: Dict[str, Any]):
  """
  Initialize the graph builder with model metadata.
  
  Args:
   metadata: Dictionary containing model metadata including layers and connections
  """
  self.metadata = metadata
  self.graph = nx.DiGraph(name=metadata.get("model_name", "Unknown Model"))
  self.layer_registry = {}
  self._connection_patterns = {}
  self._is_built = False
  self._metrics = {}
  
  logger.info(f"Initialized ArchitectureGraphBuilder for model: {self.graph.name}")

 def build_graph(self) -> nx.DiGraph:
  """
  Build the architecture graph from metadata.
  
  Returns:
   The constructed directed graph representing the model architecture
  
  Raises:
   ValueError: If the metadata format is invalid or missing required fields
  """
  try:
   if "layers" not in self.metadata:
    raise ValueError("Metadata must include 'layers' information")
    
   logger.info(f"Building architecture graph with {len(self.metadata['layers'])} layers")
   
   # Add layers as nodes
   for layer_data in self.metadata["layers"]:
    self._add_layer_node(layer_data)
   
   # Add connections as edges
   if "connections" in self.metadata:
    for connection in self.metadata["connections"]:
     self._add_connection_edge(connection)
   else:
    # If connections aren't explicitly defined, try to infer sequential connections
    self._infer_sequential_connections()
   
   # Validate the graph structure
   self._validate_graph()
   
   # Analyze the graph to identify connection patterns
   self._analyze_connection_patterns()
   
   # Calculate graph metrics
   self._calculate_metrics()
   
   self._is_built = True
   logger.info(f"Architecture graph successfully built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
   
   return self.graph
   
  except Exception as e:
   logger.error(f"Error building architecture graph: {str(e)}")
   raise

 def get_graph(self) -> nx.DiGraph:
  """
  Get the constructed architecture graph.
  
  Returns:
   The directed graph representing the model architecture
   
  Raises:
   RuntimeError: If the graph hasn't been built yet
  """
  if not self._is_built:
   logger.warning("Graph requested before being built. Building now...")
   self.build_graph()
  return self.graph

 def _add_layer_node(self, layer_data: Dict[str, Any]) -> None:
  """
  Add a layer as a node to the graph with its attributes.
  
  Args:
   layer_data: Dictionary containing layer information
  
  Raises:
   ValueError: If required layer data is missing
  """
  if "id" not in layer_data or "type" not in layer_data:
   raise ValueError("Layer data must include 'id' and 'type' fields")
   
  layer_id = layer_data["id"]
  
  # Create a standardized layer info object
  layer_info = LayerInfo(
   id=layer_id,
   name=layer_data.get("name", f"layer_{layer_id}"),
   type=layer_data["type"],
   params=layer_data.get("parameters", 0),
   input_shape=layer_data.get("input_shape"),
   output_shape=layer_data.get("output_shape"),
   activation=layer_data.get("activation"),
   trainable=layer_data.get("trainable", True)
  )
  
  # Add node with all attributes
  self.graph.add_node(
   layer_id,
   **layer_data,
   layer_info=layer_info
  )
  
  # Register layer for quick lookup
  self.layer_registry[layer_id] = layer_info
  
  logger.debug(f"Added layer node: {layer_id} ({layer_data['type']})")

 def _add_connection_edge(self, connection: Dict[str, Any]) -> None:
  """
  Add a connection as an edge to the graph with its attributes.
  
  Args:
   connection: Dictionary containing connection information
   
  Raises:
   ValueError: If required connection data is missing
  """
  if "source" not in connection or "target" not in connection:
   raise ValueError("Connection must include 'source' and 'target' fields")
   
  source = connection["source"]
  target = connection["target"]
  
  # Ensure the source and target nodes exist
  if source not in self.graph.nodes or target not in self.graph.nodes:
   missing = source if source not in self.graph.nodes else target
   logger.warning(f"Cannot add connection: node {missing} does not exist")
   return
   
  # Add the edge with all attributes
  self.graph.add_edge(
   source, 
   target,
   weight=connection.get("weight", 1.0),
   connection_type=connection.get("type", ConnectionType.SEQUENTIAL.name),
   **{k: v for k, v in connection.items() if k not in ["source", "target"]}
  )
  
  logger.debug(f"Added connection: {source} -> {target}")

 def _infer_sequential_connections(self) -> None:
  """
  Infer sequential connections between layers if not explicitly defined.
  This assumes layers are ordered in the metadata as they should be connected.
  """
  layers = self.metadata.get("layers", [])
  for i in range(len(layers) - 1):
   source_id = layers[i]["id"]
   target_id = layers[i + 1]["id"]
   
   self.graph.add_edge(
    source_id,
    target_id,
    weight=1.0,
    connection_type=ConnectionType.SEQUENTIAL.name,
    inferred=True
   )
   
  logger.info(f"Inferred {len(layers) - 1} sequential connections")

 def _validate_graph(self) -> None:
  """
  Validate the constructed graph structure.
  
  Raises:
   ValueError: If the graph structure is invalid
  """
  # Check for disconnected components
  if not nx.is_weakly_connected(self.graph) and self.graph.number_of_nodes() > 1:
   components = list(nx.weakly_connected_components(self.graph))
   logger.warning(f"Graph contains {len(components)} disconnected components")
  
  # Check for cycles (which might indicate RNNs or invalid architectures)
  cycles = list(nx.simple_cycles(self.graph))
  if cycles:
   logger.info(f"Graph contains {len(cycles)} cycles (possible recurrent connections)")
   
  # Check for input and output layers
  inputs = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
  outputs = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
  
  if not inputs:
   logger.warning("No input layers detected (no nodes with in-degree 0)")
  else:
   logger.info(f"Input layers: {inputs}")
   
  if not outputs:
   logger.warning("No output layers detected (no nodes with out-degree 0)")
  else:
   logger.info(f"Output layers: {outputs}")

 def _analyze_connection_patterns(self) -> None:
  """Analyze the graph to identify common neural network connection patterns."""
  # Find residual connections
  residual_connections = self._find_residual_connections()
  self._connection_patterns["residual"] = residual_connections
  
  # Find skip connections
  skip_connections = self._find_skip_connections()
  self._connection_patterns["skip"] = skip_connections
  
  # Find parallel branches
  parallel_branches = self._find_parallel_branches()
  self._connection_patterns["parallel"] = parallel_branches
  
  # Find merge points
  merge_points = self._find_merge_points()
  self._connection_patterns["merge"] = merge_points
  
  # Find branch points
  branch_points = self._find_branch_points()
  self._connection_patterns["branch"] = branch_points
  
  logger.info(f"Identified connection patterns: "
       f"{len(residual_connections)} residual, "
       f"{len(skip_connections)} skip, "
       f"{len(parallel_branches)} parallel branches, "
       f"{len(merge_points)} merge points, "
       f"{len(branch_points)} branch points")

 def _find_residual_connections(self) -> List[Tuple[str, str]]:
  """Identify residual connections in the architecture."""
  residuals = []
  
  # A residual connection typically has a main path with multiple layers
  # and a skip connection from an earlier layer to a later one
  for node in self.graph.nodes:
   successors = list(self.graph.successors(node))
   for successor in successors:
    # Check if there's a longer path from node to successor
    paths = list(nx.all_simple_paths(self.graph, node, successor, cutoff=5))
    if any(len(path) > 2 for path in paths):
     residuals.append((node, successor))
     
     # Mark this edge as a residual connection
     if self.graph.has_edge(node, successor):
      self.graph[node][successor]["connection_type"] = ConnectionType.RESIDUAL.name
  
  return residuals

 def _find_skip_connections(self) -> List[Tuple[str, str]]:
  """Identify skip connections in the architecture."""
  skip_connections = []
  
  # A skip connection typically jumps over at least 2 layers
  for node in self.graph.nodes:
   for target in self.graph.nodes:
    if node == target:
     continue
     
    if self.graph.has_edge(node, target):
     # Find all paths between the nodes
     paths = list(nx.all_simple_paths(self.graph, node, target, cutoff=10))
     
     # If there's another path longer than the direct edge, it's a skip connection
     longer_paths = [p for p in paths if len(p) > 2]
     if longer_paths:
      skip_connections.append((node, target))
      # Mark this edge as a skip connection
      self.graph[node][target]["connection_type"] = ConnectionType.SKIP.name
  
  return skip_connections

 def _find_parallel_branches(self) -> List[Tuple[str, List[List[str]]]]:
  """Identify parallel branches in the architecture."""
  parallel_branches = []
  
  for node in self.graph.nodes:
   out_degree = self.graph.out_degree(node)
   
   # If this node branches out to multiple nodes, it may start parallel branches
   if out_degree > 1:
    successors = list(self.graph.successors(node))
    
    # Find common endpoints for these branches
    endpoints = set()
    for successor in successors:
     # Find all nodes reachable from this successor
     reachable = nx.descendants(self.graph, successor)
     
     # Check which nodes are common endpoints for multiple branches
     for potential_endpoint in reachable:
      if potential_endpoint != successor:
       paths_to_endpoint = list(nx.all_simple_paths(self.graph, node, potential_endpoint))
       # If there are multiple paths from node to potential_endpoint
       if len(paths_to_endpoint) > 1:
        endpoints.add(potential_endpoint)
    
    if endpoints:
     for endpoint in endpoints:
      paths = list(nx.all_simple_paths(self.graph, node, endpoint))
      if len(paths) > 1 and all(len(path) > 2 for path in paths):
       parallel_branches.append((node, paths))
       
       # Mark the start node as a branch point
       self.graph.nodes[node]["is_branch_point"] = True
  
  return parallel_branches

 def _find_merge_points(self) -> List[str]:
  """Identify merge points in the architecture."""
  merge_points = []
  
  for node in self.graph.nodes:
   in_degree = self.graph.in_degree(node)
   
   # If a node has multiple inputs, it's a merge point
   if in_degree > 1:
    merge_points.append(node)
    self.graph.nodes[node]["is_merge_point"] = True
    
    # Mark the incoming edges as merge connections
    for predecessor in self.graph.predecessors(node):
     self.graph[predecessor][node]["connection_type"] = ConnectionType.MERGE.name
  
  return merge_points

 def _find_branch_points(self) -> List[str]:
  """Identify branch points in the architecture."""
  branch_points = []
  
  for node in self.graph.nodes:
   out_degree = self.graph.out_degree(node)
   
   # If a node has multiple outputs, it's a branch point
   if out_degree > 1:
    branch_points.append(node)
    self.graph.nodes[node]["is_branch_point"] = True
    
    # Mark the outgoing edges as branch connections
    for successor in self.graph.successors(node):
     self.graph[node][successor]["connection_type"] = ConnectionType.BRANCH.name
  
  return branch_points

 def _calculate_metrics(self) -> None:
  """Calculate various metrics about the architecture graph."""
  self._metrics = {
   "node_count": self.graph.number_of_nodes(),
   "edge_count": self.graph.number_of_edges(),
   "is_dag": nx.is_directed_acyclic_graph(self.graph),
   "has_cycles": not nx.is_directed_acyclic_graph(self.graph),
   "diameter": nx.diameter(self.graph.to_undirected()) if nx.is_connected(self.graph.to_undirected()) else float('inf'),
   "average_path_length": nx.average_shortest_path_length(self.graph) if nx.is_strongly_connected(self.graph) else float('nan'),
   "density": nx.density(self.graph),
   "in_degree_centrality": nx.in_degree_centrality(self.graph),
   "out_degree_centrality": nx.out_degree_centrality(self.graph),
   "is_weakly_connected": nx.is_weakly_connected(self.graph),
   "is_strongly_connected": nx.is_strongly_connected(self.graph),
   "component_count": nx.number_weakly_connected_components(self.graph),
   "bottleneck_nodes": self._identify_bottlenecks(),
   "critical_paths": self._identify_critical_paths()
  }
  
  logger.info(f"Calculated graph metrics: nodes={self._metrics['node_count']}, edges={self._metrics['edge_count']}")

 def _identify_bottlenecks(self) -> List[str]:
  """Identify bottleneck nodes in the architecture."""
  if not nx.is_weakly_connected(self.graph):
   return []
   
  # Bottlenecks are articulation points or nodes with high betweenness
  articulation_points = list(nx.articulation_points(self.graph.to_undirected()))
  
  # Calculate betweenness centrality to find nodes that many paths go through
  betweenness = nx.betweenness_centrality(self.graph)
  high_betweenness = [node for node, value in betweenness.items() 
         if value > 0.5 and node not in articulation_points]
  
  return articulation_points + high_betweenness

 def _identify_critical_paths(self) -> List[List[str]]:
  """Identify critical paths in the architecture."""
  if not nx.is_weakly_connected(self.graph):
   return []
   
  # Find input and output nodes
  inputs = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
  outputs = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
  
  critical_paths = []
  
  # For each input-output pair, find the longest path
  for input_node in inputs:
   for output_node in outputs:
    try:
     # Try to find all simple paths between input and output
     paths = list(nx.all_simple_paths(self.graph, input_node, output_node))
     if paths:
      # Sort by path length and take the longest one
      longest_path = max(paths, key=len)
      critical_paths.append(longest_path)
    except nx.NetworkXNoPath:
     continue
  
  return critical_paths

 def visualize(self, output_path: Optional[str] = None, layout: str = "spring", 
     highlight_patterns: bool = True, show_labels: bool = True) -> None:
  """
  Visualize the architecture graph.
  
  Args:
   output_path: Path to save the visualization (if None, display only)
   layout: Graph layout algorithm ('spring', 'kamada_kawai', 'circular', 'spectral')
   highlight_patterns: Whether to highlight identified connection patterns
   show_labels: Whether to show node labels
   
  Raises:
   RuntimeError: If the graph hasn't been built yet
  """
  if not self._is_built:
   raise RuntimeError("Graph must be built before visualization")
   
  plt.figure(figsize=(14, 10))
  
  # Choose layout algorithm
  if layout == "spring":
   pos = nx.spring_layout(self.graph, seed=42)
  elif layout == "kamada_kawai":
   pos = nx.kamada_kawai_layout(self.graph)
  elif layout == "circular":
   pos = nx.circular_layout(self.graph)
  elif layout == "spectral":
   pos = nx.spectral_layout(self.graph)
  else:
   logger.warning(f"Unknown layout '{layout}', using spring layout")
   pos = nx.spring_layout(self.graph, seed=42)
  
  # Prepare node colors and sizes based on layer types
  node_colors = []
  node_sizes = []
  
  for node in self.graph.nodes:
   node_type = self.graph.nodes[node].get("type", "unknown").lower()
   
   # Color by layer type
   if "conv" in node_type:
    node_colors.append("skyblue")
   elif "dense" in node_type or "linear" in node_type or "fc" in node_type:
    node_colors.append("lightgreen")
   elif "lstm" in node_type or "gru" in node_type or "rnn" in node_type:
    node_colors.append("orange")
   elif "pool" in node_type:
    node_colors.append("lightgray")
   elif "norm" in node_type:
    node_colors.append("yellow")
   elif "dropout" in node_type:
    node_colors.append("pink")
   elif "input" in node_type:
    node_colors.append("lightblue")
   elif "output" in node_type:
    node_colors.append("green")
   else:
    node_colors.append("gray")
    
   # Size by parameter count
   params = self.graph.nodes[node].get("params", 0)
   if params > 0:
    node_size = 300 + min(100 * np.log1p(params), 1000)
   else:
    node_size = 300
    
   node_sizes.append(node_size)
  
  # Draw nodes
  nodes = nx.draw_networkx_nodes(
   self.graph, pos, 
   node_color=node_colors,
   node_size=node_sizes,
   alpha=0.8
  )
  
  # Draw edges with colors based on connection type
  edge_colors = []
  edge_widths = []
  
  for source, target, data in self.graph.edges(data=True):
   conn_type = data.get("connection_type", "SEQUENTIAL")
   
   if conn_type == ConnectionType.RESIDUAL.name:
    edge_colors.append("red")
    edge_widths.append(2.0)
   elif conn_type == ConnectionType.SKIP.name:
    edge_colors.append("blue")
    edge_widths.append(1.5)
   elif conn_type == ConnectionType.MERGE.name:
    edge_colors.append("purple")
    edge_widths.append(1.8)
   elif conn_type == ConnectionType.BRANCH.name:
    edge_colors.append("green")
    edge_widths.append(1.8)
   else:
    edge_colors.append("black")
    edge_widths.append(1.0)
  
  # Draw edges
  edges = nx.draw_networkx_edges(
   self.graph, pos,
   width=edge_widths,
   edge_color=edge_colors,
   arrowstyle='-|>',
   arrowsize=15,
   connectionstyle='arc3,rad=0.1'
  )
  
  # Add labels if requested
  if show_labels:
   labels = {node: f"{node}\n({self.graph.nodes[node].get('type', 'unknown')})" 
      for node in self.graph.nodes}
   nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8)
  
  # Add a legend
  plt.legend(handles=[
   plt.Line2D([0], [0], color='skyblue', marker='o', markersize=10, label='Conv Layer', linestyle=''),
   plt.Line2D([0], [0], color='lightgreen', marker='o', markersize=10, label='Dense Layer', linestyle=''),
   plt.Line2D([0], [0], color='orange', marker='o', markersize=10, label='RNN/LSTM Layer', linestyle=''),
   plt.Line2D([0], [0], color='lightgray', marker='o', markersize=10, label='Pooling Layer', linestyle=''),
   plt.Line2D([0], [0], color='red', label='Residual Connection'),
   plt.Line2D([0], [0], color='blue', label='Skip Connection'),
   plt.Line2D([0], [0], color='purple', label='Merge Connection'),
   plt.Line2D([0], [0], color='green', label='Branch Connection'),
   plt.Line2D([0], [0], color='black', label='Sequential Connection')
  ])
  
  plt.title(f"Architecture Graph for {self.graph.name}")
  plt.axis('off')
  
  # Save the visualization if output path is provided
  if output_path:
   plt.savefig(output_path, dpi=300, bbox_inches='tight')
   logger.info(f"Saved visualization to {output_path}")
  
  plt.show()

 def get_metrics(self) -> Dict[str, Any]:
  """
  Get calculated graph metrics.
  
  Returns:
   Dictionary of graph metrics
   
  Raises:
   RuntimeError: If the graph hasn't been built yet
  """
  if not self._is_built:
   raise RuntimeError("Graph must be built before getting metrics")
  return self._metrics

 def get_connection_patterns(self) -> Dict[str, List]:
  """
  Get identified connection patterns.
  
  Returns:
   Dictionary of connection patterns
   
  Raises:
   RuntimeError: If the graph hasn't been built yet
  """
  if not self._is_built:
   raise RuntimeError("Graph must be built before getting connection patterns")
  return self._connection_patterns

 def export_to_json(self, output_path: str) -> None:
  """
  Export the graph to a JSON file.
  
  Args:
   output_path: Path to save the JSON file
   
  Raises:
   RuntimeError: If the graph hasn't been built yet
  """
  if not self._is_built:
   raise RuntimeError("Graph must be built before exporting")
   
  # Convert the graph to a dictionary
  graph_dict = {
   "name": self.graph.name,
   "metrics": self._metrics,
   "nodes": [],
   "edges": [],
   "connection_patterns": self._connection_patterns
  }
  
  # Remove non-serializable items from metrics
  graph_dict["metrics"] = {
   k: v for k, v in self._metrics.items() 
   if isinstance(v, (dict, list, str, int, float, bool)) or v is None
  }
  
  # Convert in_degree_centrality and out_degree_centrality to serializable form
  if "in_degree_centrality" in graph_dict["metrics"]:
   graph_dict["metrics"]["in_degree_centrality"] = {
    str(k): v for k, v in graph_dict["metrics"]["in_degree_centrality"].items()
   }
  if "out_degree_centrality" in graph_dict["metrics"]:
   graph_dict["metrics"]["out_degree_centrality"] = {
    str(k): v for k, v in graph_dict["metrics"]["out_degree_centrality"].items()
   }
  
  # Add all nodes with their attributes
  for node, attrs in self.graph.nodes(data=True):
   node_dict = {"id": node}
   for k, v in attrs.items():
    if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
     node_dict[k] = v
   graph_dict["nodes"].append(node_dict)
  
  # Add all edges with their attributes
  for source, target, attrs in self.graph.edges(data=True):
   edge_dict = {"source": source, "target": target}
   for k, v in attrs.items():
    if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
     edge_dict[k] = v
   graph_dict["edges"].append(edge_dict)
  
  # Convert non-serializable items in connection patterns
  serializable_patterns = {}
  for pattern_type, pattern_data in self._connection_patterns.items():
   if isinstance(pattern_data, list):
    # Convert tuples to lists
    serializable_data = []
    for item in pattern_data:
     if isinstance(item, tuple):
      serializable_data.append(list(item))
     else:
      serializable_data.append(item)
    serializable_patterns[pattern_type] = serializable_data
  
  graph_dict["connection_patterns"] = serializable_patterns
  
  # Save to JSON file
  with open(output_path, 'w') as f:
   json.dump(graph_dict, f, indent=2)
  
  logger.info(f"Exported architecture graph to {output_path}")

 @classmethod
 def from_json(cls, json_path: str) -> 'ArchitectureGraphBuilder':
  """
  Create an ArchitectureGraphBuilder instance from a JSON file.
  
  Args:
   json_path: Path to the JSON file
   
  Returns:
   ArchitectureGraphBuilder instance
   
  Raises:
   FileNotFoundError: If the JSON file doesn't exist
   ValueError: If the JSON file has an invalid format
  """
  if not Path(json_path).exists():
   raise FileNotFoundError(f"JSON file not found: {json_path}")
   
  with open(json_path, 'r') as f:
   data = json.load(f)
  
  # Create metadata from the JSON data
  metadata = {
   "model_name": data.get("name", "Imported Model"),
   "layers": data.get("nodes", []),
   "connections": data.get("edges", [])
  }
  
  # Create the builder and build the graph
  builder = cls(metadata)
  builder.build_graph()
  
  # Restore metrics and connection patterns if available
  if "metrics" in data:
   builder._metrics = data["metrics"]
  
  if "connection_patterns" in data:
   builder._connection_patterns = data["connection_patterns"]
  
  return builder

 def find_paths_between(self, source: str, target: str, max_length: int = 10) -> List[List[str]]:
  """
  Find all paths between two layers in the architecture.
  
  Args:
   source: ID of the source layer
   target: ID of the target layer
   max_length: Maximum path length to consider
   
  Returns:
   List of paths between source and target
  
  Raises:
   ValueError: If source or target node doesn't exist
   RuntimeError: If the graph hasn't been built yet
  """
  if not self._is_built:
   raise RuntimeError("Graph must be built before finding paths")
   
  if source not in self.graph.nodes:
   raise ValueError(f"Source node {source} not found in the graph")
   
  if target not in self.graph.nodes:
   raise ValueError(f"Target node {target} not found in the graph")
  
  try:
   paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
   logger.info(f"Found {len(paths)} paths between {source} and {target}")
   return paths
  except nx.NetworkXNoPath:
   logger.info(f"No path found between {source} and {target}")
   return []

 def compare_with(self, other_graph: 'ArchitectureGraphBuilder') -> Dict[str, Any]:
  """
  Compare this architecture graph with another one.
  
  Args:
   other_graph: Another ArchitectureGraphBuilder instance to compare with
   
  Returns:
   Dictionary containing comparison results
   
  Raises:
   RuntimeError: If either graph hasn't been built yet
  """
  if not self._is_built or not other_graph._is_built:
   raise RuntimeError("Both graphs must be built before comparison")
  
  this_graph = self.get_graph()
  other_g = other_graph.get_graph()
  
  comparison = {
   "node_count_diff": this_graph.number_of_nodes() - other_g.number_of_nodes(),
   "edge_count_diff": this_graph.number_of_edges() - other_g.number_of_edges(),
   "common_layer_types": self._compare_layer_types(other_graph),
   "structure_similarity": self._calculate_structure_similarity(other_graph),
   "this_unique_patterns": {},
   "other_unique_patterns": {}
  }
  
  # Compare connection patterns
  for pattern in ["residual", "skip", "parallel", "merge", "branch"]:
   this_patterns = set(map(str, self._connection_patterns.get(pattern, [])))
   other_patterns = set(map(str, other_graph._connection_patterns.get(pattern, [])))
   
   comparison["this_unique_patterns"][pattern] = len(this_patterns - other_patterns)
   comparison["other_unique_patterns"][pattern] = len(other_patterns - this_patterns)
  
  logger.info(f"Completed comparison between {this_graph.name} and {other_g.name}")
  return comparison

 def _compare_layer_types(self, other_graph: 'ArchitectureGraphBuilder') -> Dict[str, int]:
  """Compare layer type distributions between two graphs."""
  this_types = Counter([
   self.graph.nodes[node].get("type", "unknown").lower() 
   for node in self.graph.nodes
  ])
  
  other_types = Counter([
   other_graph.graph.nodes[node].get("type", "unknown").lower() 
   for node in other_graph.graph.nodes
  ])
  
  # Find common types
  common_types = {}
  all_types = set(this_types.keys()) | set(other_types.keys())
  
  for layer_type in all_types:
   common_types[layer_type] = {
    "this": this_types.get(layer_type, 0),
    "other": other_types.get(layer_type, 0),
    "diff": this_types.get(layer_type, 0) - other_types.get(layer_type, 0)
   }
  
  return common_types

 def _calculate_structure_similarity(self, other_graph: 'ArchitectureGraphBuilder') -> float:
  """Calculate a similarity score between two architecture graphs."""
  # This is a simplified similarity metric that compares graph structure
  # A more sophisticated approach would use graph edit distance or graph kernels
  
  g1 = self.graph
  g2 = other_graph.graph
  
  # Compare node counts, edge counts, density
  node_ratio = min(g1.number_of_nodes(), g2.number_of_nodes()) / max(g1.number_of_nodes(), g2.number_of_nodes())
  edge_ratio = min(g1.number_of_edges(), g2.number_of_edges()) / max(g1.number_of_edges(), g2.number_of_edges())
  
  density1 = nx.density(g1)
  density2 = nx.density(g2)
  density_ratio = min(density1, density2) / max(density1, density2)
  
  # Compare in-degree and out-degree distributions (simplified)
  in_degree1 = np.array([d for _, d in g1.in_degree()])
  in_degree2 = np.array([d for _, d in g2.in_degree()])
  
  out_degree1 = np.array([d for _, d in g1.out_degree()])
  out_degree2 = np.array([d for _, d in g2.out_degree()])
  
  # Compare mean degrees
  in_degree_ratio = min(in_degree1.mean() if len(in_degree1) > 0 else 0, 
       in_degree2.mean() if len(in_degree2) > 0 else 0) / \
        max(in_degree1.mean() if len(in_degree1) > 0 else 1, 
       in_degree2.mean() if len(in_degree2) > 0 else 1)
  
  out_degree_ratio = min(out_degree1.mean() if len(out_degree1) > 0 else 0, 
        out_degree2.mean() if len(out_degree2) > 0 else 0) / \
         max(out_degree1.mean() if len(out_degree1) > 0 else 1, 
        out_degree2.mean() if len(out_degree2) > 0 else 1)
  
  # Calculate weighted similarity score
  similarity = (
   0.3 * node_ratio +
   0.3 * edge_ratio +
   0.2 * density_ratio +
   0.1 * in_degree_ratio +
   0.1 * out_degree_ratio
  )
  
  return similarity

 def summarize(self) -> Dict[str, Any]:
  """
  Generate a summary of the architecture.
  
  Returns:
   Dictionary containing architecture summary
   
  Raises:
   RuntimeError: If the graph hasn't been built yet
  """
  if not self._is_built:
   raise RuntimeError("Graph must be built before summarizing")
   
  # Count layer types
  layer_types = Counter([
   self.graph.nodes[node].get("type", "unknown").lower() 
   for node in self.graph.nodes
  ])
  
  # Find input and output nodes
  input_nodes = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
  output_nodes = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
  
  # Calculate total parameters
  total_params = sum(self.graph.nodes[node].get("params", 0) for node in self.graph.nodes)
  
  # Calculate the maximum path length (depth)
  max_depth = 0
  for i in input_nodes:
   for o in output_nodes:
    try:
     paths = list(nx.all_simple_paths(self.graph, i, o))
     max_depth = max(max_depth, max(len(p) for p in paths) if paths else 0)
    except nx.NetworkXNoPath:
     continue
  
  # Generate the summary
  summary = {
   "name": self.graph.name,
   "total_layers": self.graph.number_of_nodes(),
   "total_connections": self.graph.number_of_edges(),
   "total_parameters": total_params,
   "layer_types": dict(layer_types),
   "input_nodes": input_nodes,
   "output_nodes": output_nodes,
   "max_depth": max_depth,
   "is_sequential": all(self.graph.out_degree(n) <= 1 for n in self.graph.nodes),
   "has_skip_connections": bool(self._connection_patterns.get("skip", [])),
   "has_residual_connections": bool(self._connection_patterns.get("residual", [])),
   "is_acyclic": nx.is_directed_acyclic_graph(self.graph),
   "connection_pattern_counts": {
    k: len(v) for k, v in self._connection_patterns.items()
   }
  }
  
  logger.info(f"Generated architecture summary for {self.graph.name}")
  return summary

 def find_subgraph_isomorphisms(self, pattern_graph: nx.DiGraph) -> List[Dict[str, str]]:
  """
  Find occurrences of a pattern graph within the architecture.
  
  Args:
   pattern_graph: A graph representing the pattern to search for
   
  Returns:
   List of dictionaries mapping pattern nodes to graph nodes
   
  Raises:
   RuntimeError: If the graph hasn't been built yet
  """
  if not self._is_built:
   raise RuntimeError("Graph must be built before finding subgraphs")
  
  try:
   # Use VF2 algorithm to find subgraph isomorphisms
   matcher = nx.algorithms.isomorphism.DiGraphMatcher(
    self.graph, pattern_graph
   )
   
   # Find all isomorphisms
   isomorphisms = list(matcher.subgraph_isomorphisms_iter())
   logger.info(f"Found {len(isomorphisms)} instances of the pattern in the architecture")
   
   # Return the isomorphisms with pattern nodes as keys and graph nodes as values
   return [{v: k for k, v in iso.items()} for iso in isomorphisms]
   
  except Exception as e:
   logger.error(f"Error finding subgraph isomorphisms: {str(e)}")
   return []

def main():
    parser = argparse.ArgumentParser(description="SVELTE Architecture Graph Builder CLI")
    parser.add_argument('model', type=str, help='Path to GGUF model file')
    args = parser.parse_args()
    from src.tensor_analysis.gguf_parser import GGUFParser
    gguf = GGUFParser(args.model)
    gguf.parse()
    from src.metadata.metadata_extractor import MetadataExtractor
    extractor = MetadataExtractor(gguf.get_metadata())
    extractor.extract()
    metadata = extractor.get_metadata()
    graph_builder = ArchitectureGraphBuilder(metadata)
    graph_builder.build_graph()
    graph = graph_builder.get_graph()
    import json
    print(json.dumps(str(graph), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
