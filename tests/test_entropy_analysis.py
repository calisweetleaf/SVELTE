# tests/test_entropy_analysis.py
"""
Unit tests for EntropyAnalysisModule in SVELTE Framework.
"""
import unittest
import numpy as np
from src.tensor_analysis.entropy_analysis import EntropyAnalysisModule

class TestEntropyAnalysisModule(unittest.TestCase):
    def test_compute_entropy(self):
        tensors = {'a': np.random.rand(100)}
        eam = EntropyAnalysisModule(tensors)
        entropy = eam.compute_entropy(bins=10)
        self.assertIn('a', entropy)
        self.assertTrue(entropy['a'] > 0)

if __name__ == "__main__":
    unittest.main()
