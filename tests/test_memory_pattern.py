# tests/test_memory_pattern.py
"""
Unit tests for MemoryPatternRecognitionSystem in SVELTE Framework.
"""
import unittest
import numpy as np
from src.model_architecture.memory_pattern import MemoryPatternRecognitionSystem

class TestMemoryPatternRecognitionSystem(unittest.TestCase):
    def test_detect_patterns(self):
        tensors = {'mem': np.random.rand(2,2)}
        mprs = MemoryPatternRecognitionSystem(tensors)
        patterns = mprs.detect_patterns()
        self.assertIn('mem', patterns)

if __name__ == "__main__":
    unittest.main()
