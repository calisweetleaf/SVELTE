# tests/test_activation_sim.py
"""
Unit tests for ActivationSpaceSimulator in SVELTE Framework.
"""
import unittest
import numpy as np
from src.tensor_analysis.activation_sim import ActivationSpaceSimulator

class TestActivationSpaceSimulator(unittest.TestCase):
    def test_simulate(self):
        tensors = {'a': np.random.rand(2,2)}
        sim = ActivationSpaceSimulator(tensors)
        dummy_input = np.random.rand(1,2)
        activations = sim.simulate(dummy_input)
        self.assertIsInstance(activations, dict)

if __name__ == "__main__":
    unittest.main()
