# tests/test_attention_topology.py
"""
Unit tests for AttentionTopologySystem in SVELTE Framework.
"""
import unittest
import numpy as np
from src.model_architecture.attention_topology import AttentionTopologySystem

class TestAttentionTopologySystem(unittest.TestCase):
    def test_compute_curvature(self):
        tensors = {'attn': np.random.rand(2,2)}
        ats = AttentionTopologySystem(tensors)
        curv = ats.compute_curvature()
        self.assertIn('attn', curv)

if __name__ == "__main__":
    unittest.main()
