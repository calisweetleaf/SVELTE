"""Simplified metadata extraction utilities for tests."""

from __future__ import annotations

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Lightweight metadata extractor used in unit tests."""

    def __init__(self, source: Dict[str, Any] | str) -> None:
        if isinstance(source, dict):
            self.file_path: Optional[str] = None
            self.raw_metadata = source
        else:
            self.file_path = source
            self.raw_metadata = {}
        self.metadata: Dict[str, Any] = {}

    def extract(self) -> Dict[str, Any]:
        """Extract metadata from the provided source."""
        if self.file_path:
            # In this simplified implementation we just record the file path
            self.metadata = {"source": self.file_path}
        else:
            self.metadata = dict(self.raw_metadata)
        logger.info("Metadata extraction finished")
        return self.metadata

    def get_metadata(self) -> Dict[str, Any]:
        """Return extracted metadata."""
        return self.metadata
