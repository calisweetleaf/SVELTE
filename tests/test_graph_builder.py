# tests/test_graph_builder.py
"""
Unit tests for ArchitectureGraphBuilder in SVELTE Framework.
"""
import unittest
from src.model_architecture.graph_builder import ArchitectureGraphBuilder

class TestArchitectureGraphBuilder(unittest.TestCase):
    def test_build_graph(self):
        meta = {'layers': ['a', 'b']}
        builder = ArchitectureGraphBuilder(meta)
        builder.build_graph()
        graph = builder.get_graph()
        self.assertIsNotNone(graph)

if __name__ == "__main__":
    unittest.main()
