# tests/test_tensor_field.py
"""
Unit tests for TensorFieldConstructor in SVELTE Framework.
"""
import unittest
import numpy as np
from src.tensor_analysis.tensor_field import TensorFieldConstructor

class TestTensorFieldConstructor(unittest.TestCase):
    def test_construct(self):
        tensors = {'a': [[1,2],[3,4]], 'b': [5,6,7]}
        tfc = TensorFieldConstructor(tensors)
        field = tfc.construct()
        self.assertTrue('a' in field and 'b' in field)
        self.assertTrue(isinstance(field['a'], np.ndarray))

if __name__ == "__main__":
    unittest.main()
