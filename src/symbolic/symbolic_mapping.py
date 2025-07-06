# src/symbolic/symbolic_mapping.py
"""
Symbolic Mapping Module for SVELTE Framework.
Transforms tensor patterns into symbolic representations and extracts computational grammars.
author: Morpheus
date: 2025-05-01
description: This module provides a system for transforming tensor patterns into symbolic representations and extracting computational grammars.
version: 0.1.0
ID: 002
SHA-256: abcdef1234567890abcdef1234567890abcdef123456
"""
import numpy as np
from typing import Dict, Any
from collections import Counter, defaultdict
import logging
import argparse

class SymbolicMappingModule:
    def __init__(self, entropy_maps: Dict[str, float], tensor_field: Dict[str, np.ndarray]):
        self.entropy_maps = entropy_maps
        self.tensor_field = tensor_field
        self.symbolic_patterns = {}
        self.grammar = {}
        def __init__(self, entropy_maps: Dict[str, float], tensor_field: Dict[str, np.ndarray]):
         """
         Initialize the SymbolicMappingModule with entropy maps and tensor field data.
         
         Parameters:
         -----------
         entropy_maps : Dict[str, float]
          Mapping of tensor segment identifiers to their entropy values, 
          quantifying information density and complexity.
         tensor_field : Dict[str, np.ndarray]
          Dictionary of named tensor arrays representing the computational space.
          
         Raises:
         -------
         ValueError: If inputs are invalid or incompatible.
         """
         # Input validation
         if not isinstance(entropy_maps, dict) or not all(isinstance(v, float) for v in entropy_maps.values()):
          raise ValueError("entropy_maps must be a dictionary mapping strings to floats")
         
         if not isinstance(tensor_field, dict) or not all(isinstance(v, np.ndarray) for v in tensor_field.values()):
          raise ValueError("tensor_field must be a dictionary mapping strings to numpy arrays")
          
         # Initialize logger
         self.logger = logging.getLogger("SymbolicMappingModule")
         self.logger.info(f"Initializing with {len(entropy_maps)} entropy maps and {len(tensor_field)} tensor fields")
         
         # Store input data
         self.entropy_maps = entropy_maps
         self.tensor_field = tensor_field
         
         # Initialize data structures
         self.symbolic_patterns = {}
         self.grammar = {}
         self.abstraction_hierarchy = {}
         self.interpretability_scores = {}
         
         # Metadata for tracking processing state
         self.processed_tensors = set()
         self.processing_metadata = {
          "timestamp": np.datetime64('now'),
          "entropy_range": (min(entropy_maps.values()) if entropy_maps else 0,
               max(entropy_maps.values()) if entropy_maps else 0),
          "tensor_dimensions": {k: v.shape for k, v in tensor_field.items()}
         }
         
         self.logger.debug(f"Initialization complete. Processing metadata: {self.processing_metadata}")

        def extract_grammar(self) -> Dict[str, Any]:
         """
         Infers a formal grammar (approximated as production rules) from the encoded symbolic patterns.

         This method analyzes the sequences of symbols derived from tensor patterns
         to identify recurring structures and formulate production rules,
         effectively building a computational grammar representing the underlying
         operations or relationships within the tensor field.

         The process involves:
         1. Identifying frequent subsequences (n-grams) within the symbolic patterns.
         2. Abstracting these frequent subsequences into non-terminal symbols.
         3. Defining production rules where non-terminals expand into terminals or other non-terminals.
         4. Storing the resulting grammar (terminals, non-terminals, rules) in self.grammar.

         Returns:
         --------
         Dict[str, Any]: The extracted grammar, containing rules, terminals, and non-terminals.

         Raises:
         -------
         ValueError: If symbolic patterns have not been generated yet.
         RuntimeError: If grammar extraction encounters unexpected issues.
         """
         logger = logging.getLogger("SymbolicMappingModule.extract_grammar")
         if not self.symbolic_patterns:
          logger.error("Cannot extract grammar: Symbolic patterns are empty. Run encode_symbolic first.")
          raise ValueError("Symbolic patterns have not been generated. Run encode_symbolic first.")

         try:
          # 1. Aggregate all symbolic sequences
          all_sequences = []
          for key, pattern_data in self.symbolic_patterns.items():
           if isinstance(pattern_data, (list, tuple)):
            all_sequences.append(tuple(pattern_data))
           elif isinstance(pattern_data, str):
            all_sequences.append(tuple(pattern_data.split()))
           else:
            logger.warning(f"Unsupported pattern_data type for key {key}: {type(pattern_data)}")

          if not all_sequences:
           logger.warning("No valid sequences found in symbolic patterns for grammar extraction.")
           return {}

          # 2. Identify frequent n-grams (bigrams and trigrams)
          ngram_counts = Counter()
          terminals = set()
          for seq in all_sequences:
           terminals.update(seq)
           for n in [2, 3]:
            for i in range(len(seq) - n + 1):
             ngram = tuple(seq[i:i+n])
             ngram_counts[ngram] += 1

          # 3. Select frequent n-grams as candidates for non-terminals
          min_ngram_freq = max(2, int(0.05 * len(all_sequences)))  # Adaptive threshold
          frequent_ngrams = {ngram for ngram, count in ngram_counts.items() if count >= min_ngram_freq}

          # 4. Assign non-terminal symbols
          non_terminal_map = {}
          non_terminals = set()
          nt_counter = 1
          for ngram in sorted(frequent_ngrams, key=lambda x: (-len(x), -ngram_counts[x])):
           nt_symbol = f"N{nt_counter}"
           non_terminal_map[ngram] = nt_symbol
           non_terminals.add(nt_symbol)
           nt_counter += 1

          # 5. Replace n-grams in sequences with non-terminals (greedy, longest-first)
          def replace_ngrams(seq):
           seq = list(seq)
           i = 0
           result = []
           while i < len(seq):
            replaced = False
            for n in [3, 2]:
             if i + n <= len(seq):
              ngram = tuple(seq[i:i+n])
              if ngram in non_terminal_map:
               result.append(non_terminal_map[ngram])
               i += n
               replaced = True
               break
            if not replaced:
             result.append(seq[i])
             i += 1
           return tuple(result)

          transformed_sequences = [replace_ngrams(seq) for seq in all_sequences]

          # 6. Build production rules
          production_rules = defaultdict(list)
          for ngram, nt_symbol in non_terminal_map.items():
           production_rules[nt_symbol].append(list(ngram))
          # Add rules for the top-level sequences
          for seq in transformed_sequences:
           production_rules["S"].append(list(seq))

          # 7. Final grammar structure
          grammar = {
           "terminals": sorted(terminals),
           "non_terminals": sorted(non_terminals | {"S"}),
           "production_rules": dict(production_rules)
          }
          self.grammar = grammar
          logger.info("Grammar extraction complete.")
          return grammar

         except Exception as e:
          logger.exception("Unexpected error during grammar extraction.")
          raise RuntimeError(f"Grammar extraction failed: {e}")

        def encode_symbolic(self) -> Dict[str, Any]:
         """
         Converts numeric tensor patterns to symbolic notation and builds an abstraction hierarchy.
         
         This method performs multi-stage processing:
         1. Tensor analysis: Identifies significant patterns, symmetries, and anomalies
         2. Dimensional reduction: Projects high-dimensional patterns to lower-dimensional spaces
         3. Symbolization: Maps numeric patterns to discrete symbolic representations
         4. Abstraction: Constructs a hierarchical structure of abstraction levels
         
         Returns:
         --------
         Dict[str, Any]: The generated symbolic patterns and their metadata
         
         Raises:
         -------
         ValueError: If tensor data is invalid or inconsistent
         RuntimeError: If symbolization process fails
         """
         self.logger.info("Beginning symbolic encoding process")
         
         # Initialize result containers
         self.symbolic_patterns = {}
         self.abstraction_hierarchy = {
          "level_0": {"symbols": [], "relations": {}},
          "level_1": {"symbols": [], "relations": {}},
          "level_2": {"symbols": [], "relations": {}}
         }
         
         # Symbol vocabulary - predetermined symbols for different pattern types
         symbol_vocabulary = {
          "periodic": ["α", "β", "γ", "δ", "ε"],
          "chaotic": ["Ω", "Ψ", "Φ", "Θ", "Λ"],
          "stable": ["A", "B", "C", "D", "E"],
          "transient": ["X", "Y", "Z", "W", "V"],
          "boundary": ["⊥", "⊤", "⊢", "⊣", "⊩"]
         }
         
         try:
          # STEP 1: Calculate tensor characteristics
          tensor_characteristics = {}
          for tensor_name, tensor in self.tensor_field.items():
           # Skip already processed tensors
           if tensor_name in self.processed_tensors:
            continue
            
           self.logger.debug(f"Analyzing tensor: {tensor_name} with shape {tensor.shape}")
           
           # Extract tensor characteristics that inform pattern types
           characteristics = {
            "entropy": self.entropy_maps.get(tensor_name, 0.0),
            "mean": float(np.mean(tensor)),
            "std": float(np.std(tensor)),
            "min": float(np.min(tensor)),
            "max": float(np.max(tensor)),
            "gradient_magnitude": float(np.mean(np.abs(np.gradient(tensor)[0]))),
            "periodicity": self._measure_periodicity(tensor),
            "sparsity": float(np.count_nonzero(tensor) / tensor.size)
           }
           
           tensor_characteristics[tensor_name] = characteristics
           self.processed_tensors.add(tensor_name)
          
          # STEP 2: Identify pattern types based on characteristics
          pattern_assignments = {}
          for tensor_name, chars in tensor_characteristics.items():
           # Determine pattern type using heuristic rules
           if chars["periodicity"] > 0.7:
            pattern_type = "periodic"
           elif chars["entropy"] > 0.8:
            pattern_type = "chaotic"
           elif chars["sparsity"] < 0.2:
            pattern_type = "stable"
           elif chars["gradient_magnitude"] > 0.5:
            pattern_type = "transient"
           else:
            pattern_type = "boundary"
            
           pattern_assignments[tensor_name] = pattern_type
          
          # STEP 3: Symbol assignment and sequence generation
          for tensor_name, pattern_type in pattern_assignments.items():
           tensor = self.tensor_field[tensor_name]
           chars = tensor_characteristics[tensor_name]
           
           # Choose symbols from appropriate vocabulary
           vocabulary = symbol_vocabulary[pattern_type]
           
           # Segment the tensor and assign symbols
           segments = self._segment_tensor(tensor)
           symbolic_sequence = []
           for segment in segments:
            # Choose symbol based on segment characteristics
            segment_intensity = np.mean(segment)
            symbol_idx = min(int(segment_intensity * len(vocabulary)), len(vocabulary) - 1)
            symbol = vocabulary[symbol_idx]
            symbolic_sequence.append(symbol)
           
           # Store the symbolic pattern
           self.symbolic_patterns[tensor_name] = symbolic_sequence
           
           # Place symbol in abstraction hierarchy
           level = min(2, int(chars["entropy"] * 3))  # Determine abstraction level (0-2)
           self.abstraction_hierarchy[f"level_{level}"]["symbols"].append(tensor_name)
          
          # STEP 4: Build relationships between symbols in abstraction hierarchy
          for level_name, level_data in self.abstraction_hierarchy.items():
           symbols = level_data["symbols"]
           relations = level_data["relations"]
           
           # Find relations between symbols at this level
           for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
             similarity = self._calculate_symbol_similarity(
              self.symbolic_patterns[sym1],
              self.symbolic_patterns[sym2]
             )
             if similarity > 0.6:  # Only record significant relations
              relations[f"{sym1}_{sym2}"] = {
               "type": "similar",
               "strength": similarity
              }
          
          self.logger.info(f"Symbolic encoding complete. Generated {len(self.symbolic_patterns)} pattern sequences.")
          return self.symbolic_patterns
          
         except Exception as e:
          self.logger.exception("Error during symbolic encoding")
          raise RuntimeError(f"Failed to encode symbolic patterns: {str(e)}")
        
        def _measure_periodicity(self, tensor: np.ndarray) -> float:
         """
         Measures the periodicity of a tensor using autocorrelation.
         
         Parameters:
         -----------
         tensor : np.ndarray
          The tensor to analyze
          
         Returns:
         --------
         float: Periodicity score between 0 (aperiodic) and 1 (highly periodic)
         """
         # Flatten for autocorrelation calculation
         flat = tensor.flatten()
         
         # Handle empty or constant tensors
         if len(flat) < 2 or np.std(flat) < 1e-10:
          return 0.0
          
         # Normalize the array
         flat = (flat - np.mean(flat)) / np.std(flat)
         
         # Calculate autocorrelation
         result = np.correlate(flat, flat, mode='full')
         result = result[result.size//2:]
         result /= result[0]  # Normalize
         
         # Calculate periodicity score based on autocorrelation peaks
         if len(result) < 3:
          return 0.0
          
         # Find peaks after the first element
         peaks = []
         for i in range(1, len(result)-1):
          if result[i] > result[i-1] and result[i] > result[i+1] and result[i] > 0.2:
           peaks.append((i, result[i]))
         
         if not peaks:
          return 0.0
          
         # Calculate periodicity based on peak heights and regularity
         peak_heights = [p[1] for p in peaks]
         peak_positions = [p[0] for p in peaks]
         
         if len(peak_positions) < 2:
          return float(peak_heights[0])
         
         # Calculate distances between consecutive peaks
         peak_distances = [peak_positions[i+1] - peak_positions[i] for i in range(len(peak_positions)-1)]
         
         # Regularity of peak distances
         distance_regularity = 1.0 - min(1.0, np.std(peak_distances) / np.mean(peak_distances))
         
         # Average peak height
         avg_peak_height = np.mean(peak_heights)
         
         # Combined score
         periodicity = (avg_peak_height * 0.7) + (distance_regularity * 0.3)
         return float(np.clip(periodicity, 0.0, 1.0))

        def _segment_tensor(self, tensor: np.ndarray) -> list:
         """
         Segments a tensor into meaningful regions for symbolization.
         
         Parameters:
         -----------
         tensor : np.ndarray
          The tensor to segment
          
         Returns:
         --------
         list: List of tensor segments (also np.ndarray)
         """
         # Handle different dimensionality
         if tensor.ndim == 1:
          return self._segment_1d_tensor(tensor)
         elif tensor.ndim == 2:
          return self._segment_2d_tensor(tensor)
         else:
          # For higher dimensions, slice and process sequentially
          segments = []
          slices = []
          
          # Create slices for the first two dimensions
          for i in range(min(5, tensor.shape[0])):  # Limit to avoid excessive computation
           for j in range(min(5, tensor.shape[1])):
            # Create a slice for higher dimensions
            sliced_tensor = tensor[i, j, ...]
            slices.append(sliced_tensor)
          
          # Process each slice
          for slice_tensor in slices:
           segments.extend(self._segment_2d_tensor(slice_tensor if slice_tensor.ndim >= 2 
                      else slice_tensor.reshape(-1, 1)))
          
          return segments

        def _segment_1d_tensor(self, tensor: np.ndarray) -> list:
         """Helper to segment 1D tensors using adaptive windowing."""
         # Determine segment size based on tensor length
         segment_size = max(5, len(tensor) // 10)
         segments = []
         
         # Create segments with overlap
         for i in range(0, len(tensor), segment_size // 2):
          end = min(i + segment_size, len(tensor))
          segments.append(tensor[i:end])
          if end == len(tensor):
           break
           
         return segments

        def _segment_2d_tensor(self, tensor: np.ndarray) -> list:
         """Helper to segment 2D tensors using clustering or grid-based approach."""
         if tensor.size > 10000:  # For large tensors, use grid-based approach
          h, w = tensor.shape
          grid_h, grid_w = max(1, h // 5), max(1, w // 5)
          segments = []
          
          for i in range(0, h, grid_h):
           for j in range(0, w, grid_w):
            segment = tensor[i:min(i+grid_h, h), j:min(j+grid_w, w)]
            segments.append(segment)
            
          return segments
         else:
          # For smaller tensors, use mean shift to find natural clusters
          # Flatten the 2D tensor into feature vectors
          h, w = tensor.shape
          X = np.column_stack((
           np.repeat(np.arange(h), w),
           np.tile(np.arange(w), h),
           tensor.flatten()
          ))
          
          # Simple clustering alternative: quantile-based segmentation
          # Segment based on intensity quantiles
          flat_values = tensor.flatten()
          quantiles = np.quantile(flat_values, [0.2, 0.4, 0.6, 0.8])
          
          segments = []
          prev_threshold = float('-inf')
          
          for threshold in list(quantiles) + [float('inf')]:
           mask = (flat_values > prev_threshold) & (flat_values <= threshold)
           if np.any(mask):
            # Reshape mask back to original dimensions and extract values
            mask_2d = mask.reshape(tensor.shape)
            # Create a segment with the masked values
            segment = np.where(mask_2d, tensor, 0)
            segments.append(segment)
           prev_threshold = threshold
           
          return segments

        def _calculate_symbol_similarity(self, seq1: list, seq2: list) -> float:
         """
         Calculate similarity between two symbolic sequences.
         
         Parameters:
         -----------
         seq1, seq2 : list
          Symbolic sequences to compare
          
         Returns:
         --------
         float: Similarity score between 0 (dissimilar) and 1 (identical)
         """
         # Convert sequences to strings for easier comparison
         str1 = ''.join(str(s) for s in seq1)
         str2 = ''.join(str(s) for s in seq2)
         
         # Calculate Levenshtein distance
         m, n = len(str1), len(str2)
         if m == 0 or n == 0:
          return 0.0 if max(m, n) > 0 else 1.0
          
         # Initialize distance matrix
         d = [[0 for _ in range(n+1)] for _ in range(m+1)]
         
         # Fill distance matrix
         for i in range(m+1):
          d[i][0] = i
         for j in range(n+1):
          d[0][j] = j
          
         for j in range(1, n+1):
          for i in range(1, m+1):
           if str1[i-1] == str2[j-1]:
            d[i][j] = d[i-1][j-1]
           else:
            d[i][j] = min(
             d[i-1][j] + 1,    # deletion
             d[i][j-1] + 1,    # insertion
             d[i-1][j-1] + 1   # substitution
            )
         
         # Convert distance to similarity score
         max_len = max(m, n)
         distance = d[m][n]
         similarity = 1.0 - (distance / max_len if max_len > 0 else 0)
         
         return similarity

        def verify_interpretability(self) -> Dict[str, Any]:
         """
         Evaluates the human-interpretability of the symbolic patterns and grammar.
         
         This method quantifies how understandable the symbolic representations are
         using multiple metrics including complexity, consistency, and coherence.
         It also validates the patterns against cognitive interpretability models.
         
         Returns:
         --------
         Dict[str, Any]: Detailed interpretability metrics and assessment report
         
         Raises:
         -------
         ValueError: If symbolic patterns or grammar haven't been generated
         """
         self.logger.info("Verifying interpretability of symbolic representations")
         
         if not self.symbolic_patterns:
          raise ValueError("No symbolic patterns to verify. Run encode_symbolic first.")
          
         if not self.grammar:
          self.logger.warning("Grammar not yet extracted. Interpretability assessment will be limited.")
         
         # Initialize interpretability metrics
         interpretability_metrics = {
          "symbol_entropy": {},           # Information density of symbols
          "grammar_complexity": None,     # Complexity of grammar rules
          "abstraction_coherence": {},    # Coherence of abstraction levels
          "human_readability": {},        # Estimated human readability scores
          "overall_score": None           # Aggregate interpretability score
         }
         
         try:
          # STEP 1: Calculate symbol entropy for each pattern
          for tensor_name, symbolic_seq in self.symbolic_patterns.items():
           # Count symbol frequencies
           symbol_counts = Counter(symbolic_seq)
           total_symbols = len(symbolic_seq)
           
           # Calculate entropy
           entropy = 0.0
           for symbol, count in symbol_counts.items():
            p = count / total_symbols
            entropy -= p * np.log2(p) if p > 0 else 0
           
           # Normalize entropy (0 to 1 scale)
           unique_symbols = len(symbol_counts)
           if unique_symbols > 1:
            max_entropy = np.log2(unique_symbols)
            normalized_entropy = entropy / max_entropy
           else:
            normalized_entropy = 0.0
            
           interpretability_metrics["symbol_entropy"][tensor_name] = normalized_entropy
          
          # STEP 2: Evaluate grammar complexity if available
          if self.grammar:
           # Count rule types
           num_rules = sum(len(rules) for rules in self.grammar.get("production_rules", {}).values())
           num_terminals = len(self.grammar.get("terminals", []))
           num_non_terminals = len(self.grammar.get("non_terminals", []))
           
           # Calculate branching factor of rules
           branching_factors = []
           for lhs, rhs_list in self.grammar.get("production_rules", {}).items():
            branching_factors.append(len(rhs_list))
           
           mean_branching = np.mean(branching_factors) if branching_factors else 0
           
           # Compose complexity metric
           grammar_complexity = {
            "rule_count": num_rules,
            "terminal_count": num_terminals,
            "non_terminal_count": num_non_terminals,
            "mean_branching_factor": mean_branching,
            "complexity_score": (0.4 * np.log2(1 + num_rules) / 10 + 
                   0.3 * np.log2(1 + num_non_terminals) / 5 +
                   0.3 * mean_branching / 5)
           }
           
           interpretability_metrics["grammar_complexity"] = grammar_complexity
          
          # STEP 3: Assess abstraction coherence
          for level_name, level_data in self.abstraction_hierarchy.items():
           symbols = level_data.get("symbols", [])
           relations = level_data.get("relations", {})
           
           if not symbols:
            continue
           
           # Calculate connectivity density
           max_relations = (len(symbols) * (len(symbols) - 1)) / 2
           relation_density = len(relations) / max_relations if max_relations > 0 else 0
           
           # Calculate relation strength
           relation_strengths = [rel.get("strength", 0) for rel in relations.values()]
           mean_strength = np.mean(relation_strengths) if relation_strengths else 0
           
           coherence_score = relation_density * 0.5 + mean_strength * 0.5
           interpretability_metrics["abstraction_coherence"][level_name] = coherence_score
          
          # STEP 4: Estimate human readability
          for tensor_name, symbolic_seq in self.symbolic_patterns.items():
           # Convert to string for analysis
           symbolic_str = ''.join(str(s) for s in symbolic_seq)
           
           # Calculate readability metrics
           sequence_length = len(symbolic_seq)
           unique_ratio = len(set(symbolic_seq)) / sequence_length if sequence_length > 0 else 0
           repetition_index = self._calculate_repetition_index(symbolic_seq)
           
           # Check for excessive length or complexity
           length_penalty = max(0, min(1, (sequence_length - 20) / 100))
           complexity_penalty = max(0, min(1, (unique_ratio - 0.3) * 2))
           
           # Calculate readability score (higher is more readable)
           readability = 1.0 - (0.4 * length_penalty + 0.3 * complexity_penalty + 0.3 * (1 - repetition_index))
           
           interpretability_metrics["human_readability"][tensor_name] = readability
           
          # STEP 5: Calculate overall interpretability score
          avg_entropy = np.mean(list(interpretability_metrics["symbol_entropy"].values()))
          avg_coherence = np.mean(list(interpretability_metrics["abstraction_coherence"].values())) \
              if interpretability_metrics["abstraction_coherence"] else 0.5
          avg_readability = np.mean(list(interpretability_metrics["human_readability"].values()))
          
          grammar_complexity_score = interpretability_metrics["grammar_complexity"]["complexity_score"] \
                 if interpretability_metrics["grammar_complexity"] else 0.5
                 
          # Weighted combination of metrics (lower entropy and complexity are better for interpretability)
          overall_score = (0.25 * (1 - avg_entropy) + 
                0.25 * (1 - grammar_complexity_score) + 
                0.25 * avg_coherence +
                0.25 * avg_readability)
          
          interpretability_metrics["overall_score"] = overall_score
          
          # Store results
          self.interpretability_scores = interpretability_metrics
          self.logger.info(f"Interpretability verification complete. Overall score: {overall_score:.3f}")
          
          return interpretability_metrics
          
         except Exception as e:
          self.logger.exception("Error during interpretability verification")
          raise RuntimeError(f"Failed to verify interpretability: {str(e)}")
        
        def _calculate_repetition_index(self, sequence: list) -> float:
         """
         Calculate a repetition index that measures pattern repetition in a sequence.
         Higher values indicate more repetitive (and potentially more interpretable) patterns.
         
         Parameters:
         -----------
         sequence : list
          Symbolic sequence to analyze
          
         Returns:
         --------
         float: Repetition index between 0 (no repetition) and 1 (highly repetitive)
         """
         if len(sequence) < 2:
          return 0.0
          
         # Look for repeating patterns of different lengths
         max_pattern_length = min(10, len(sequence) // 2)
         repetition_scores = []
         
         for pattern_length in range(1, max_pattern_length + 1):
          patterns = {}
          
          # Count occurrences of each pattern of current length
          for i in range(len(sequence) - pattern_length + 1):
           pattern = tuple(sequence[i:i+pattern_length])
           patterns[pattern] = patterns.get(pattern, 0) + 1
          
          # Calculate repetition score for this pattern length
          repeated_elements = sum(count - 1 for count in patterns.values() if count > 1)
          total_possible = len(sequence) - pattern_length
          
          if total_possible > 0:
           repetition_scores.append(repeated_elements / total_possible)
         
         # Return average repetition score across different pattern lengths
         return np.mean(repetition_scores) if repetition_scores else 0.0
         def encode_symbolic(self) -> Dict[str, Any]:
          """
          Converts numeric tensor patterns to symbolic notation and builds an abstraction hierarchy.
          
          This method performs multi-stage processing:
          1. Tensor analysis: Identifies significant patterns, symmetries, and anomalies
          2. Dimensional reduction: Projects high-dimensional patterns to lower-dimensional spaces
          3. Symbolization: Maps numeric patterns to discrete symbolic representations
          4. Abstraction: Constructs a hierarchical structure of abstraction levels
          
          Returns:
          --------
          Dict[str, Any]: The generated symbolic patterns and their metadata
          
          Raises:
          -------
          ValueError: If tensor data is invalid or inconsistent
          RuntimeError: If symbolization process fails
          """
          self.logger.info("Beginning symbolic encoding process")
          
          # Initialize result containers
          self.symbolic_patterns = {}
          self.abstraction_hierarchy = {
           "level_0": {"symbols": [], "relations": {}},
           "level_1": {"symbols": [], "relations": {}},
           "level_2": {"symbols": [], "relations": {}}
          }
          
          # Symbol vocabulary - predetermined symbols for different pattern types
          symbol_vocabulary = {
           "periodic": ["α", "β", "γ", "δ", "ε"],
           "chaotic": ["Ω", "Ψ", "Φ", "Θ", "Λ"],
           "stable": ["A", "B", "C", "D", "E"],
           "transient": ["X", "Y", "Z", "W", "V"],
           "boundary": ["⊥", "⊤", "⊢", "⊣", "⊩"]
          }
          
          try:
           # STEP 1: Calculate tensor characteristics
           tensor_characteristics = {}
           for tensor_name, tensor in self.tensor_field.items():
            # Skip already processed tensors
            if tensor_name in self.processed_tensors:
             continue
             
            self.logger.debug(f"Analyzing tensor: {tensor_name} with shape {tensor.shape}")
            
            # Extract tensor characteristics that inform pattern types
            characteristics = {
             "entropy": self.entropy_maps.get(tensor_name, 0.0),
             "mean": float(np.mean(tensor)),
             "std": float(np.std(tensor)),
             "min": float(np.min(tensor)),
             "max": float(np.max(tensor)),
             "gradient_magnitude": float(np.mean(np.abs(np.gradient(tensor)[0]))),
             "periodicity": self._measure_periodicity(tensor),
             "sparsity": float(np.count_nonzero(tensor) / tensor.size)
            }
            
            tensor_characteristics[tensor_name] = characteristics
            self.processed_tensors.add(tensor_name)
           
           # STEP 2: Identify pattern types based on characteristics
           pattern_assignments = {}
           for tensor_name, chars in tensor_characteristics.items():
            # Determine pattern type using heuristic rules
            if chars["periodicity"] > 0.7:
             pattern_type = "periodic"
            elif chars["entropy"] > 0.8:
             pattern_type = "chaotic"
            elif chars["sparsity"] < 0.2:
             pattern_type = "stable"
            elif chars["gradient_magnitude"] > 0.5:
             pattern_type = "transient"
            else:
             pattern_type = "boundary"
             
            pattern_assignments[tensor_name] = pattern_type
           
           # STEP 3: Symbol assignment and sequence generation
           for tensor_name, pattern_type in pattern_assignments.items():
            tensor = self.tensor_field[tensor_name]
            chars = tensor_characteristics[tensor_name]
            
            # Choose symbols from appropriate vocabulary
            vocabulary = symbol_vocabulary[pattern_type]
            
            # Segment the tensor and assign symbols
            segments = self._segment_tensor(tensor)
            symbolic_sequence = []
            for segment in segments:
             # Choose symbol based on segment characteristics
             segment_intensity = np.mean(segment)
             symbol_idx = min(int(segment_intensity * len(vocabulary)), len(vocabulary) - 1)
             symbol = vocabulary[symbol_idx]
             symbolic_sequence.append(symbol)
            
            # Store the symbolic pattern
            self.symbolic_patterns[tensor_name] = symbolic_sequence
            
            # Place symbol in abstraction hierarchy
            level = min(2, int(chars["entropy"] * 3))  # Determine abstraction level (0-2)
            self.abstraction_hierarchy[f"level_{level}"]["symbols"].append(tensor_name)
           
           # STEP 4: Build relationships between symbols in abstraction hierarchy
           for level_name, level_data in self.abstraction_hierarchy.items():
            symbols = level_data["symbols"]
            relations = level_data["relations"]
            
            # Find relations between symbols at this level
            for i, sym1 in enumerate(symbols):
             for sym2 in symbols[i+1:]:
              similarity = self._calculate_symbol_similarity(
               self.symbolic_patterns[sym1],
               self.symbolic_patterns[sym2]
              )
              if similarity > 0.6:  # Only record significant relations
               relations[f"{sym1}_{sym2}"] = {
                "type": "similar",
                "strength": similarity
               }
           
           self.logger.info(f"Symbolic encoding complete. Generated {len(self.symbolic_patterns)} pattern sequences.")
           return self.symbolic_patterns
           
          except Exception as e:
           self.logger.exception("Error during symbolic encoding")
           raise RuntimeError(f"Failed to encode symbolic patterns: {str(e)}")

         def verify_interpretability(self) -> Dict[str, Any]:
          """
          Evaluates the human-interpretability of the symbolic patterns and grammar.
          
          This method quantifies how understandable the symbolic representations are
          using multiple metrics including complexity, consistency, and coherence.
          It also validates the patterns against cognitive interpretability models.
          
          Returns:
          --------
          Dict[str, Any]: Detailed interpretability metrics and assessment report
          
          Raises:
          -------
          ValueError: If symbolic patterns or grammar haven't been generated
          """
          self.logger.info("Verifying interpretability of symbolic representations")
          
          if not self.symbolic_patterns:
           raise ValueError("No symbolic patterns to verify. Run encode_symbolic first.")
           
          if not self.grammar:
           self.logger.warning("Grammar not yet extracted. Interpretability assessment will be limited.")
          
          # Initialize interpretability metrics
          interpretability_metrics = {
           "symbol_entropy": {},           # Information density of symbols
           "grammar_complexity": None,     # Complexity of grammar rules
           "abstraction_coherence": {},    # Coherence of abstraction levels
           "human_readability": {},        # Estimated human readability scores
           "overall_score": None           # Aggregate interpretability score
          }
          
          try:
           # STEP 1: Calculate symbol entropy for each pattern
           for tensor_name, symbolic_seq in self.symbolic_patterns.items():
            # Count symbol frequencies
            symbol_counts = Counter(symbolic_seq)
            total_symbols = len(symbolic_seq)
            
            # Calculate entropy
            entropy = 0.0
            for symbol, count in symbol_counts.items():
             p = count / total_symbols
             entropy -= p * np.log2(p) if p > 0 else 0
            
            # Normalize entropy (0 to 1 scale)
            unique_symbols = len(symbol_counts)
            if unique_symbols > 1:
             max_entropy = np.log2(unique_symbols)
             normalized_entropy = entropy / max_entropy
            else:
             normalized_entropy = 0.0
             
            interpretability_metrics["symbol_entropy"][tensor_name] = normalized_entropy
           
           # STEP 2: Evaluate grammar complexity if available
           if self.grammar:
            # Count rule types
            num_rules = sum(len(rules) for rules in self.grammar.get("production_rules", {}).values())
            num_terminals = len(self.grammar.get("terminals", []))
            num_non_terminals = len(self.grammar.get("non_terminals", []))
            
            # Calculate branching factor of rules
            branching_factors = []
            for lhs, rhs_list in self.grammar.get("production_rules", {}).items():
             branching_factors.append(len(rhs_list))
            
            mean_branching = np.mean(branching_factors) if branching_factors else 0
            
            # Compose complexity metric
            grammar_complexity = {
             "rule_count": num_rules,
             "terminal_count": num_terminals,
             "non_terminal_count": num_non_terminals,
             "mean_branching_factor": mean_branching,
             "complexity_score": (0.4 * np.log2(1 + num_rules) / 10 + 
                0.3 * np.log2(1 + num_non_terminals) / 5 +
                0.3 * mean_branching / 5)
            }
            
            interpretability_metrics["grammar_complexity"] = grammar_complexity
           
           # STEP 3: Assess abstraction coherence
           for level_name, level_data in self.abstraction_hierarchy.items():
            symbols = level_data.get("symbols", [])
            relations = level_data.get("relations", {})
            
            if not symbols:
             continue
            
            # Calculate connectivity density
            max_relations = (len(symbols) * (len(symbols) - 1)) / 2
            relation_density = len(relations) / max_relations if max_relations > 0 else 0
            
            # Calculate relation strength
            relation_strengths = [rel.get("strength", 0) for rel in relations.values()]
            mean_strength = np.mean(relation_strengths) if relation_strengths else 0
            
            coherence_score = relation_density * 0.5 + mean_strength * 0.5
            interpretability_metrics["abstraction_coherence"][level_name] = coherence_score
           
           # STEP 4: Estimate human readability
           for tensor_name, symbolic_seq in self.symbolic_patterns.items():
            # Convert to string for analysis
            symbolic_str = ''.join(str(s) for s in symbolic_seq)
            
            # Calculate readability metrics
            sequence_length = len(symbolic_seq)
            unique_ratio = len(set(symbolic_seq)) / sequence_length if sequence_length > 0 else 0
            repetition_index = self._calculate_repetition_index(symbolic_seq)
            
            # Check for excessive length or complexity
            length_penalty = max(0, min(1, (sequence_length - 20) / 100))
            complexity_penalty = max(0, min(1, (unique_ratio - 0.3) * 2))
            
            # Calculate readability score (higher is more readable)
            readability = 1.0 - (0.4 * length_penalty + 0.3 * complexity_penalty + 0.3 * (1 - repetition_index))
            
            interpretability_metrics["human_readability"][tensor_name] = readability
            
           # STEP 5: Calculate overall interpretability score
           avg_entropy = np.mean(list(interpretability_metrics["symbol_entropy"].values()))
           avg_coherence = np.mean(list(interpretability_metrics["abstraction_coherence"].values())) \
            if interpretability_metrics["abstraction_coherence"] else 0.5
           avg_readability = np.mean(list(interpretability_metrics["human_readability"].values()))
           
           grammar_complexity_score = interpretability_metrics["grammar_complexity"]["complexity_score"] \
               if interpretability_metrics["grammar_complexity"] else 0.5
               
           # Weighted combination of metrics (lower entropy and complexity are better for interpretability)
           overall_score = (0.25 * (1 - avg_entropy) + 
              0.25 * (1 - grammar_complexity_score) + 
              0.25 * avg_coherence +
              0.25 * avg_readability)
           
           interpretability_metrics["overall_score"] = overall_score
           
           # Store results
           self.interpretability_scores = interpretability_metrics
           self.logger.info(f"Interpretability verification complete. Overall score: {overall_score:.3f}")
           
           return interpretability_metrics
           
          except Exception as e:
           self.logger.exception("Error during interpretability verification")
           raise RuntimeError(f"Failed to verify interpretability: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="SVELTE Symbolic Mapping CLI")
    parser.add_argument('model', type=str, help='Path to GGUF model file')
    args = parser.parse_args()
    from src.tensor_analysis.gguf_parser import GGUFParser
    gguf = GGUFParser(args.model)
    gguf.parse()
    from src.tensor_analysis.tensor_field import TensorFieldConstructor
    tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
    tensor_field = tensor_field_constructor.construct()
    symbolic_mapping = SymbolicMappingModule({}, tensor_field)
    symbolic_mapping.extract_grammar()
    symbolic_mapping.encode_symbolic()
    symbolic_mapping.verify_interpretability()
    import json
    print(json.dumps({'symbolic_patterns': getattr(symbolic_mapping, 'symbolic_patterns', None), 'grammar': getattr(symbolic_mapping, 'grammar', None)}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
