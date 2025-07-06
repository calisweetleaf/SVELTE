import pytest
import numpy as np
import networkx as nx
import logging
from src.symbolic.meta_interpretation import (
    MetaInterpretationSynthesisModule,
    InterpretationNode,
    InterpretationLevel,
    TaxonomicClass,
    InterpretationGraph,
    ConflictResolutionStrategy
)

# --- Fixtures ---

@pytest.fixture
def minimal_module_outputs():
    """Minimal mock module outputs for basic testing."""
    return {
        "module_A": {
            "findings": [
                {"description": "Pattern X observed", "confidence": 0.8, "details": {"value": 10}},
                {"description": "Feature Y prominent", "confidence": 0.7, "details": {"value": 20}}
            ]
        },
        "module_B": {
            "results": [ # Using a different key to test generic processor
                {"description": "Connection Z strong", "confidence": 0.9, "strength": 5.5}
            ]
        }
    }

@pytest.fixture
def attention_module_output():
    return {
        "topology_metrics": type('obj', (object,), {
            'curvature': np.array([0.1, 0.5, 1.2, 0.8, 2.5]), # Example curvature values
            'eigenvalues': np.array([3.0, 2.5, 1.0, 0.5, 0.1]) # Example eigenvalues
        })()
    }

@pytest.fixture
def memory_module_output():
    # Mocking PatternType directly if not easily importable or for simplicity
    MockPatternType = type('Enum', (object,), {'value': 'recurrent', 'name':'RECURRENT'})
    return {
        "memory_motifs": [
            type('obj', (object,), {
                'name': "motif1", 'layer_ids': ["L1", "L2"], 'pattern_type': MockPatternType,
                'strength': 0.85, 'frequency': 5, 'variance': 0.1, 'connections': set(["motif2"])
            })()
        ]
    }

@pytest.fixture
def graph_module_output():
    mock_graph = nx.DiGraph()
    mock_graph.add_node("L1", type="Dense")
    mock_graph.add_node("L2", type="Conv")
    mock_graph.add_edge("L1", "L2")
    return {"graph": mock_graph}


@pytest.fixture
def mism_instance(minimal_module_outputs):
    """Basic MISM instance."""
    return MetaInterpretationSynthesisModule(minimal_module_outputs)

# --- InterpretationNode Tests ---
def test_interpretation_node_creation():
    node = InterpretationNode(
        id="test_node_1",
        source_module="test_module",
        description="A test finding.",
        level=InterpretationLevel.FUNCTIONAL,
        confidence=0.75,
        evidence=[{"type": "data_point", "value": 42}]
    )
    assert node.id == "test_node_1"
    assert node.confidence == 0.75
    assert node.dependencies == set()
    assert node.conflicts == set()

# --- InterpretationGraph Tests ---
def test_interpretation_graph_add_node():
    graph = InterpretationGraph()
    node1 = InterpretationNode("n1", "modA", "desc1", InterpretationLevel.PRIMITIVE, 0.8, [])
    graph.add_node(node1)
    assert "n1" in graph.graph
    assert graph.node_index["n1"] == node1

def test_interpretation_graph_add_edge():
    graph = InterpretationGraph()
    node1 = InterpretationNode("n1", "modA", "d1", InterpretationLevel.PRIMITIVE, 0.8, [])
    node2 = InterpretationNode("n2", "modB", "d2", InterpretationLevel.FUNCTIONAL, 0.7, [])
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge("n1", "n2", "supports", weight=0.9)
    assert graph.graph.has_edge("n1", "n2")
    assert graph.graph.edges["n1", "n2"]["type"] == "supports"

# --- MetaInterpretationSynthesisModule Tests ---
class TestMetaInterpretationSynthesisModule:

    def test_initialization(self, minimal_module_outputs):
        mism = MetaInterpretationSynthesisModule(minimal_module_outputs)
        assert mism.module_outputs == minimal_module_outputs
        assert mism.conflict_strategy == ConflictResolutionStrategy.EVIDENCE_BASED # Default
        assert mism.logger is not None

    def test_generic_interpretation_processor(self, mism_instance):
        # Test with module_B which should use the generic processor
        generic_output = minimal_module_outputs()["module_B"]
        interpretations = mism_instance._generic_interpretation_processor("module_B", generic_output)
        assert len(interpretations) == 1
        interp = interpretations[0]
        assert interp.id == "module_B_results_0"
        assert interp.source_module == "module_B"
        assert interp.description == "Connection Z strong"
        assert interp.confidence == 0.9

    def test_process_attention_topology(self, mism_instance, attention_module_output):
        interpretations = mism_instance._process_attention_topology(attention_module_output)
        assert len(interpretations) > 0 # Expect interpretations from curvature and eigenvalues
        for interp in interpretations:
            assert interp.source_module == "attention_topology"
            assert interp.level == InterpretationLevel.FUNCTIONAL

    def test_process_memory_patterns(self, mism_instance, memory_module_output):
        interpretations = mism_instance._process_memory_patterns(memory_module_output)
        assert len(interpretations) == 1
        assert interpretations[0].id == "mem_motif_0"
        assert interpretations[0].description.startswith("Recurrent memory pattern")

    def test_process_architecture_graph(self, mism_instance, graph_module_output):
        # This processor looks for hubs, components, bottlenecks.
        # The provided graph_module_output is too simple to trigger all.
        interpretations = mism_instance._process_architecture_graph(graph_module_output)
        # For a 2-node graph, it might not find significant hubs or large components.
        # This tests that it runs without error.
        assert isinstance(interpretations, list)


    def test_synthesize_runs_with_minimal_data(self, mism_instance, caplog):
        """Test that the main synthesize method runs without crashing on minimal data."""
        caplog.set_level(logging.INFO)
        # Ensure some patterns are added to the graph for other steps to not be trivial
        node1 = InterpretationNode("n1", "modA", "desc1", InterpretationLevel.PRIMITIVE, 0.8, [])
        node2 = InterpretationNode("n2", "modB", "desc2", InterpretationLevel.FUNCTIONAL, 0.7, [])
        mism_instance.interpretation_graph.add_node(node1)
        mism_instance.interpretation_graph.add_node(node2)

        results = mism_instance.synthesize()
        assert isinstance(results, dict)
        assert "summary" in results
        assert "abstraction_levels" in results
        assert "taxonomy" in results
        assert "key_findings" in results
        assert "INFO:Beginning meta-interpretation synthesis process" in caplog.text
        assert "INFO:Synthesis complete" in caplog.text


    def test_synthesize_with_specific_module_outputs(self, attention_module_output, memory_module_output, graph_module_output):
        """Test synthesize with more specific module outputs."""
        # This is a more integrated test.
        # It primarily checks that the pipeline runs.
        # Detailed assertion of the output content would be very complex.

        # Create a MISM instance with outputs from specific modules.
        # The processors for these modules should be called.
        combined_outputs = {
            "attention_topology": attention_module_output,
            "memory_pattern": memory_module_output,
            "architecture_graph": graph_module_output # Renamed for clarity
        }
        mism = MetaInterpretationSynthesisModule(combined_outputs)

        results = mism.synthesize()
        assert "summary" in results
        assert len(mism.interpretation_graph.node_index) > 0 # Should have some nodes from processors

    def test_conflict_resolution_confidence(self):
        outputs = {} # Not used directly by this specific test part
        mism = MetaInterpretationSynthesisModule(outputs, conflict_strategy=ConflictResolutionStrategy.CONFIDENCE_WEIGHTED)
        node1 = InterpretationNode("n1", "mod", "d1", InterpretationLevel.FUNCTIONAL, 0.9, [])
        node2 = InterpretationNode("n2", "mod", "d2_conflicting", InterpretationLevel.FUNCTIONAL, 0.7, [])
        node1.conflicts.add("n2") # Manually set conflict for test
        node2.conflicts.add("n1")

        mism.interpretation_graph.add_node(node1)
        mism.interpretation_graph.add_node(node2)

        resolved = mism._detect_and_resolve_conflicts()
        assert len(resolved) > 0 # Might be 1 or 2 depending on how find_conflicts and iteration work
        # Check that one of the resolutions selected n1
        assert any("Selected n1" in res_tuple[2] for res_tuple in resolved)

    def test_build_abstraction_hierarchy(self, mism_instance):
        # Add some nodes at different levels
        p1 = InterpretationNode("p1", "m", "prim1", InterpretationLevel.PRIMITIVE, 0.8, [{"type":"val"}])
        f1 = InterpretationNode("f1", "m", "func1 based on prim1", InterpretationLevel.FUNCTIONAL, 0.7, [])
        b1 = InterpretationNode("b1", "m", "behav1 based on func1", InterpretationLevel.BEHAVIORAL, 0.9, [])
        mism_instance.interpretation_graph.add_node(p1)
        mism_instance.interpretation_graph.add_node(f1)
        mism_instance.interpretation_graph.add_node(b1)

        hierarchy = mism_instance.build_abstraction_hierarchy()
        assert InterpretationLevel.PRIMITIVE in hierarchy
        assert InterpretationLevel.FUNCTIONAL in hierarchy
        assert hierarchy[InterpretationLevel.PRIMITIVE].has_node("p1")
        # Check if relationships were built (heuristic, might not always link)
        # This part is hard to test reliably without exact control over _find_supporting_nodes
        # For now, just check execution and basic structure.
        assert mism_instance.interpretation_graph.graph.has_edge("p1", "f1") or \
               mism_instance.interpretation_graph.graph.has_edge("f1", "b1") # At least one link expected by heuristics

    def test_generate_taxonomy(self, mism_instance):
        # Add some functional nodes for taxonomy generation
        f1 = InterpretationNode("f1", "m", "attention head variant alpha", InterpretationLevel.FUNCTIONAL, 0.8, [])
        f2 = InterpretationNode("f2", "m", "attention head variant beta", InterpretationLevel.FUNCTIONAL, 0.85, [])
        f3 = InterpretationNode("f3", "m", "feedforward block type gamma", InterpretationLevel.FUNCTIONAL, 0.7, [])
        mism_instance.interpretation_graph.add_node(f1)
        mism_instance.interpretation_graph.add_node(f2)
        mism_instance.interpretation_graph.add_node(f3)

        taxonomy = mism_instance.generate_taxonomy()
        assert isinstance(taxonomy, dict)
        # Expecting f1 and f2 to potentially cluster if their feature vectors are similar enough
        # The feature vectors are very simple currently (confidence, evidence count etc.)
        # This test ensures it runs and produces some classes.
        if taxonomy:
            assert len(taxonomy) > 0
            first_class_name = list(taxonomy.keys())[0]
            assert isinstance(taxonomy[first_class_name], TaxonomicClass)

    def test_export_to_json(self, mism_instance, tmp_path):
        mism_instance.synthesize() # Populate integrated_interpretation
        output_file = tmp_path / "meta_interpretation.json"
        mism_instance.export_to_json(str(output_file))
        assert output_file.exists()
        with open(output_file, 'r') as f:
            data = json.load(f)
        assert "summary" in data
        assert "overall_confidence" in data["summary"] # Check one key from _generate_summary

print("Created test file: tests/symbolic/test_meta_interpretation.py")
