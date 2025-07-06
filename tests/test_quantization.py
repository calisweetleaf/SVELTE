# tests/test_quantization.py
"""
Unit tests for QuantizationReconstructor in SVELTE Framework.
"""
import unittest
import numpy as np
from src.tensor_analysis.quantization import QuantizationReconstructor

class TestQuantizationReconstructor(unittest.TestCase):
    def test_simulate_dequantization(self):
        recon = QuantizationReconstructor({'scheme': 'int8'})
        tensor = np.array([1,2,3])
        out = recon.simulate_dequantization(tensor)
        self.assertTrue((out == tensor).all())
    def test_identify_artifacts(self):
        recon = QuantizationReconstructor({'scheme': 'int8'})
        tensor = np.array([1,2,3])
        artifacts = recon.identify_artifacts(tensor)
        self.assertIsInstance(artifacts, dict)

if __name__ == "__main__":
    unittest.main()
