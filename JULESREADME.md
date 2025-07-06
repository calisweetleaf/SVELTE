# SVELTE Framework - Jules AI Agent Instructions

## Project Overview

The **SVELTE Framework** (Symbolic Vector Entropy & Latent Tensor Excavation) is a comprehensive system for analyzing and interpreting large language model architectures through differential geometry, entropy analysis, and symbolic mapping.

## Current Implementation Status

This codebase has a solid foundation with most architectural components in place, but several modules need completion of placeholder implementations to be production-ready.

## Key Tasks for Jules

### 1. Complete Missing Implementations in Core Modules

#### `src/model_architecture/attention_topology.py`

- **Current Status**: Has imports and class structure but missing method implementations
- **What to implement**: Complete the missing methods in `AttentionTopologySystem` class
- **Key methods needed**:
  - `_setup_logging()` - Configure logging for the module
  - `_validate_inputs()` - Validate tensor field inputs
  - `compute_curvature()` - Calculate manifold curvature tensors
  - `analyze_topology()` - Main topology analysis method
  - `visualize_curvature()` - Visualization method for curvature data
- **Dependencies**: numpy, scipy, matplotlib (already in imports)

#### `src/model_architecture/graph_builder.py`

- **Current Status**: Partially implemented
- **What to implement**: Complete graph construction and visualization methods
- **Focus**: NetworkX graph building from tensor relationships

#### `src/model_architecture/memory_pattern.py`

- **Current Status**: Has structure but needs pattern recognition algorithms
- **What to implement**: Memory pattern detection and analysis methods

#### `src/symbolic/meta_interpretation.py`

- **Current Status**: Framework present but core synthesis methods incomplete
- **What to implement**: Meta-interpretation synthesis algorithms

#### `src/symbolic/symbolic_mapping.py`

- **Current Status**: Basic structure exists
- **What to implement**: Pattern transformation and symbolic encoding methods

### 2. Implement Missing Modules

#### `src/tensor_analysis/` modules

- Complete any placeholder implementations in:
  - `entropy_analysis.py`
  - `tensor_field.py`
  - `activation_sim.py`
  - `quantization.py`

#### `src/utils/file_io.py`

- Implement file I/O utilities for GGUF and tensor data handling

### 3. Create Production-Ready CLI Interface

#### `main.py`

- **Current Status**: Basic structure exists
- **What to implement**: Complete command-line interface with proper argument parsing
- **Features needed**:
  - Model file input handling
  - Analysis mode selection
  - Output configuration
  - Progress reporting
  - Error handling

### 4. Implement Missing Framework Components

#### Cognitive Cartography Module

- **Location**: `src/cognitive_cartography/`
- **What to create**: Centralized visualization system as described in framework docs
- **Files needed**:
  - `visualization_engine.py`
  - `interactive_interface.py`

#### Expansion Systems

- **Location**: `src/expansion_systems/`
- **What to implement**: Advanced analysis capabilities
- **Files to complete**:
  - `cross_model_comparison.py`
  - `counterfactual_analysis.py`
  - `regulatory_compliance.py`

#### Governing Systems

- **Location**: `src/governing_systems/`
- **What to implement**: Framework control and validation
- **Files to complete**:
  - `theoretical_foundation.py`
  - `operational_control.py`

## Implementation Guidelines

### Code Quality Standards

- Follow the existing code style and patterns
- Add comprehensive docstrings for all methods
- Include proper error handling and logging
- Add input validation for all public methods
- Use type hints throughout

### Testing Requirements

- Ensure all test files in `tests/` directory pass
- Add any missing test coverage for new implementations
- Test files follow the pattern: `test_*.py`

### Dependencies

- Use only the dependencies already specified in the project
- Primary dependencies: numpy, scipy, matplotlib, networkx
- All imports are already configured in existing files

### Documentation

- Update docstrings to match implementation
- Ensure all methods have proper documentation
- Follow the existing documentation style

## Architecture Notes

### Framework Design

The SVELTE framework follows a modular architecture:

1. **Tensor Analysis**: Core numerical processing
2. **Model Architecture**: Graph and topology analysis
3. **Symbolic Mapping**: Pattern transformation
4. **Cognitive Cartography**: Visualization
5. **Metadata Extraction**: Model information processing

### Key Patterns

- All modules follow similar initialization patterns
- Error handling uses custom exception classes
- Logging is configured per module
- Results are returned as structured dictionaries

## Expected Outputs

After implementation, the framework should:

1. Successfully parse GGUF model files
2. Perform entropy and topology analysis
3. Generate symbolic mappings
4. Create visualizations of model architecture
5. Produce comprehensive analysis reports

## Files to Focus On (Priority Order)

1. `src/model_architecture/attention_topology.py` - Core topology analysis
2. `src/tensor_analysis/entropy_analysis.py` - Entropy calculations
3. `src/symbolic/symbolic_mapping.py` - Pattern transformation
4. `main.py` - CLI interface
5. `src/cognitive_cartography/visualization_engine.py` - Visualization system

## Testing Strategy

- Run `python -m pytest tests/` to validate implementations
- All tests should pass after implementation
- Focus on the test files to understand expected behavior

## Notes for Jules

- The codebase has excellent structure and documentation
- Focus on completing the TODO/placeholder implementations
- Maintain consistency with existing patterns and style
- The framework design is solid - just need the implementations
- All necessary imports and dependencies are already configured
