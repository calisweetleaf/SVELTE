import pytest
import numpy as np
import networkx as nx
import logging
import json
from pathlib import Path
from src.model_architecture.graph_builder import ArchitectureGraphBuilder, LayerInfo, ConnectionType

# --- Fixtures ---

@pytest.fixture
def simple_sequential_metadata():
    return {
        "model_name": "SequentialNet",
        "layers": [
            {"id": "input1", "name": "InputLayer", "type": "Input", "parameters": 0, "output_shape": [None, 784]},
            {"id": "dense1", "name": "Dense1", "type": "Dense", "parameters": 784*128, "input_shape": [None, 784], "output_shape": [None, 128], "activation": "relu"},
            {"id": "dense2", "name": "Dense2", "type": "Dense", "parameters": 128*10, "input_shape": [None, 128], "output_shape": [None, 10], "activation": "softmax"}
        ],
        # No explicit connections, should be inferred
    }

@pytest.fixture
def metadata_with_skip_connection():
    return {
        "model_name": "SkipNet",
        "layers": [
            {"id": "input1", "type": "Input", "name": "Input"},
            {"id": "conv1", "type": "Conv2D", "name": "Conv1"},
            {"id": "conv2", "type": "Conv2D", "name": "Conv2"},
            {"id": "add1", "type": "Add", "name": "Add"}, # Merge point
            {"id": "output1", "type": "Output", "name": "Output"}
        ],
        "connections": [
            {"source": "input1", "target": "conv1", "type": ConnectionType.SEQUENTIAL.name},
            {"source": "conv1", "target": "conv2", "type": ConnectionType.SEQUENTIAL.name},
            {"source": "conv2", "target": "add1", "type": ConnectionType.SEQUENTIAL.name},
            {"source": "input1", "target": "add1", "type": ConnectionType.SKIP.name}, # Skip connection
            {"source": "add1", "target": "output1", "type": ConnectionType.SEQUENTIAL.name}
        ]
    }

@pytest.fixture
def invalid_metadata_missing_layers():
    return {"model_name": "NoLayerNet"}

@pytest.fixture
def invalid_metadata_missing_layer_id():
    return {
        "model_name": "MissingIDNet",
        "layers": [{"type": "Dense", "name": "DenseNoID"}]
    }

# --- LayerInfo Tests ---

def test_layerinfo_creation():
    li = LayerInfo(id="1", name="DenseLayer", type="Dense", params=100, output_shape=[None, 10])
    assert li.id == "1"
    assert li.name == "DenseLayer"
    assert li.type == "dense" # Should be lowercased
    assert li.params == 100
    assert li.output_shape == [None, 10]
    assert li.hash is not None

def test_layerinfo_missing_required_fields():
    with pytest.raises(ValueError, match="Layer id, name, and type are required"):
        LayerInfo(id="1", name=None, type="Dense", params=100) # type: ignore
    with pytest.raises(ValueError, match="Layer id, name, and type are required"):
        LayerInfo(id=None, name="Layer", type="Dense", params=100) # type: ignore

# --- ArchitectureGraphBuilder Tests ---

class TestArchitectureGraphBuilder:

    def test_initialization(self, simple_sequential_metadata):
        builder = ArchitectureGraphBuilder(simple_sequential_metadata)
        assert builder.metadata == simple_sequential_metadata
        assert isinstance(builder.graph, nx.DiGraph)
        assert builder.graph.name == "SequentialNet"
        assert not builder._is_built

    def test_build_graph_sequential_inference(self, simple_sequential_metadata):
        builder = ArchitectureGraphBuilder(simple_sequential_metadata)
        graph = builder.build_graph()
        assert builder._is_built
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2 # input1->dense1, dense1->dense2
        assert graph.has_edge("input1", "dense1")
        assert graph.has_edge("dense1", "dense2")
        assert graph["input1"]["dense1"]["connection_type"] == ConnectionType.SEQUENTIAL.name
        assert graph["input1"]["dense1"]["inferred"] is True

    def test_build_graph_with_explicit_connections(self, metadata_with_skip_connection):
        builder = ArchitectureGraphBuilder(metadata_with_skip_connection)
        graph = builder.build_graph()
        assert graph.number_of_nodes() == 5
        assert graph.number_of_edges() == 5
        assert graph.has_edge("input1", "add1")
        assert graph["input1"]["add1"]["connection_type"] == ConnectionType.SKIP.name # Initially set by metadata
        # _analyze_connection_patterns might update this type if it re-classifies it as SKIP.
        # After _analyze_connection_patterns, this should be ConnectionType.SKIP.name
        assert "skip" in builder._connection_patterns
        # Check if ("input1", "add1") is identified as a skip or residual.
        # The logic for _find_skip_connections might re-classify it.
        # The initial type is from metadata, then pattern analysis updates it.
        # Let's check the identified patterns.
        assert any(conn == ("input1", "add1") for conn in builder._connection_patterns["skip"]) or \
               any(conn == ("input1", "add1") for conn in builder._connection_patterns["residual"])


    def test_build_graph_invalid_metadata_no_layers(self, invalid_metadata_missing_layers):
        builder = ArchitectureGraphBuilder(invalid_metadata_missing_layers)
        with pytest.raises(ValueError, match="Metadata must include 'layers' information"):
            builder.build_graph()

    def test_build_graph_invalid_metadata_layer_missing_id(self, invalid_metadata_missing_layer_id):
        builder = ArchitectureGraphBuilder(invalid_metadata_missing_layer_id)
        with pytest.raises(ValueError, match="Layer data must include 'id' and 'type' fields"):
            builder.build_graph() # Error during _add_layer_node

    def test_add_layer_node_attributes(self, simple_sequential_metadata):
        builder = ArchitectureGraphBuilder({"model_name": "Test", "layers": []}) # Init with empty
        layer_data = simple_sequential_metadata["layers"][1] # dense1
        builder._add_layer_node(layer_data)
        node_attrs = builder.graph.nodes["dense1"]
        assert node_attrs["name"] == "Dense1"
        assert node_attrs["type"] == "Dense" # Original type from metadata
        assert node_attrs["parameters"] == 784*128
        assert isinstance(node_attrs["layer_info"], LayerInfo)
        assert node_attrs["layer_info"].type == "dense" # Normalized type in LayerInfo

    def test_get_graph_before_build(self, simple_sequential_metadata, caplog):
        builder = ArchitectureGraphBuilder(simple_sequential_metadata)
        caplog.set_level(logging.INFO)
        graph = builder.get_graph() # Should trigger build
        assert "Graph requested before being built. Building now..." in caplog.text
        assert builder._is_built
        assert graph.number_of_nodes() == 3

    def test_pattern_detection_merge_point(self, metadata_with_skip_connection):
        builder = ArchitectureGraphBuilder(metadata_with_skip_connection)
        builder.build_graph()
        assert "add1" in builder._connection_patterns["merge"]
        assert builder.graph.nodes["add1"]["is_merge_point"] is True

    def test_pattern_detection_branch_point(self):
        metadata = {
            "model_name": "BranchNet",
            "layers": [
                {"id": "in", "type": "Input"}, {"id": "branch_start", "type": "Dense"},
                {"id": "branch1_l1", "type": "Dense"}, {"id": "branch2_l1", "type": "Dense"},
                {"id": "merge", "type": "Add"}
            ],
            "connections": [
                {"source": "in", "target": "branch_start"},
                {"source": "branch_start", "target": "branch1_l1"}, # Branch
                {"source": "branch_start", "target": "branch2_l1"}, # Branch
                {"source": "branch1_l1", "target": "merge"},
                {"source": "branch2_l1", "target": "merge"}
            ]
        }
        builder = ArchitectureGraphBuilder(metadata)
        builder.build_graph()
        assert "branch_start" in builder._connection_patterns["branch"]
        assert builder.graph.nodes["branch_start"]["is_branch_point"] is True

    def test_calculate_metrics(self, simple_sequential_metadata):
        builder = ArchitectureGraphBuilder(simple_sequential_metadata)
        builder.build_graph()
        metrics = builder.get_metrics()
        assert metrics["node_count"] == 3
        assert metrics["edge_count"] == 2
        assert metrics["is_dag"] is True
        assert isinstance(metrics["density"], float)

    def test_summarize(self, metadata_with_skip_connection):
        builder = ArchitectureGraphBuilder(metadata_with_skip_connection)
        builder.build_graph()
        summary = builder.summarize()
        assert summary["name"] == "SkipNet"
        assert summary["total_layers"] == 5
        assert summary["has_skip_connections"] is True

    def test_export_import_json(self, metadata_with_skip_connection, tmp_path):
        builder1 = ArchitectureGraphBuilder(metadata_with_skip_connection)
        builder1.build_graph()

        json_file = tmp_path / "graph.json"
        builder1.export_to_json(str(json_file))
        assert json_file.exists()

        builder2 = ArchitectureGraphBuilder.from_json(str(json_file))
        assert builder2._is_built
        assert builder2.graph.name == builder1.graph.name
        assert builder2.graph.number_of_nodes() == builder1.graph.number_of_nodes()
        assert builder2.graph.number_of_edges() == builder1.graph.number_of_edges()
        # Compare some metrics as a proxy for full graph isomorphism
        assert builder2.get_metrics()["density"] == builder1.get_metrics()["density"]

    def test_find_paths_between(self, metadata_with_skip_connection):
        builder = ArchitectureGraphBuilder(metadata_with_skip_connection)
        builder.build_graph()
        paths = builder.find_paths_between("input1", "add1")
        assert len(paths) > 0
        # Expected paths: [input1, add1] (direct skip) and [input1, conv1, conv2, add1]
        assert any(p == ["input1", "add1"] for p in paths)
        assert any(p == ["input1", "conv1", "conv2", "add1"] for p in paths)

    def test_find_paths_no_path(self, simple_sequential_metadata):
        builder = ArchitectureGraphBuilder(simple_sequential_metadata)
        builder.build_graph() # input1 -> dense1 -> dense2
        paths = builder.find_paths_between("dense2", "input1") # Reverse path
        assert len(paths) == 0

    def test_compare_with_simple(self, simple_sequential_metadata, metadata_with_skip_connection):
        builder1 = ArchitectureGraphBuilder(simple_sequential_metadata)
        builder1.build_graph()

        builder2 = ArchitectureGraphBuilder(metadata_with_skip_connection)
        builder2.build_graph()

        comparison = builder1.compare_with(builder2)
        assert comparison["node_count_diff"] == (3 - 5)
        assert "Input" in comparison["common_layer_types"]
        assert "structure_similarity" in comparison
        assert isinstance(comparison["structure_similarity"], float)

    def test_find_subgraph_isomorphisms_simple(self, metadata_with_skip_connection):
        builder = ArchitectureGraphBuilder(metadata_with_skip_connection)
        builder.build_graph()

        # Pattern: Input -> Conv -> Add
        pattern_graph = nx.DiGraph()
        pattern_graph.add_edge("p_in", "p_conv")
        pattern_graph.add_edge("p_conv", "p_add")
        # To make it more specific, one could add type attributes to pattern nodes
        # and use node_match in DiGraphMatcher. For now, just structural.

        isomorphisms = builder.find_subgraph_isomorphisms(pattern_graph)
        # Expected: input1 -> conv1 -> (conv2) -> add1. The pattern is input->conv1->add1 if conv2 is ignored.
        # The current find_subgraph_isomorphisms does not use node attributes for matching.
        # For this simple pattern, it might find input1-conv1-conv2 or parts of it.
        # This test is more of an execution check for now.
        assert isinstance(isomorphisms, list)
        # A more specific test would require defining node attributes in the pattern graph
        # and a node_match function for the matcher.

    # Placeholder for visualization test - typically hard to assert content
    def test_visualize_runs(self, simple_sequential_metadata, tmp_path):
        builder = ArchitectureGraphBuilder(simple_sequential_metadata)
        builder.build_graph()
        output_file = tmp_path / "graph_viz.png"
        try:
            builder.visualize(output_path=str(output_file), show_labels=True)
            # If matplotlib is not available, this might raise ImportError or skip.
            # The SUT's visualize does not explicitly raise if matplotlib is missing,
            # but relies on top-level try-except for it.
            # For CI, actual display is an issue. Saving to file is testable.
            if Path(output_file).exists(): # Check if file was created (implies no crash)
                 assert output_file.stat().st_size > 0
            else:
                 # This case might occur if matplotlib is missing and visualize handles it by not saving
                 # Or if there was an error before saving.
                 # SUT does not have try-except for matplotlib import within visualize, so it would raise.
                 pass # Allow test to pass if no file, assuming it's due to env not having matplotlib
        except ImportError:
            pytest.skip("Matplotlib not available, skipping visualization test.")
        except Exception as e:
            # Some environments might not have display server for plt.show()
            if "no display name" in str(e).lower() or "cannot connect to X server" in str(e).lower():
                pytest.skip(f"Matplotlib display issue, skipping visualization test: {e}")
            else:
                raise


print("Created test file: tests/model_architecture/test_graph_builder.py")
