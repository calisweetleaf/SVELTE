# tests/test_metadata_extractor.py
"""
Unit tests for MetadataExtractor in SVELTE Framework.
"""
import unittest
from src.metadata.metadata_extractor import MetadataExtractor

class TestMetadataExtractor(unittest.TestCase):
    def test_extract(self):
        raw = {'arch': 'test', 'vocab': ['a', 'b']}
        extractor = MetadataExtractor(raw)
        extractor.extract()
        self.assertIsInstance(extractor.get_metadata(), dict)

if __name__ == "__main__":
    unittest.main()
