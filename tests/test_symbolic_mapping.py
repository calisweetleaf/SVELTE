# tests/test_symbolic_mapping.py
"""
Unit tests for SymbolicMappingModule in SVELTE Framework.
"""
import unittest
import numpy as np
from src.symbolic.symbolic_mapping import SymbolicMappingModule

class TestSymbolicMappingModule(unittest.TestCase):
    def test_init(self):
        entropy = {'a': 1.0}
        tensors = {'a': np.array([1,2,3])}
        smm = SymbolicMappingModule(entropy, tensors)
        self.assertEqual(smm.entropy_maps, entropy)
        self.assertEqual(smm.tensor_field, tensors)

if __name__ == "__main__":
    unittest.main()
