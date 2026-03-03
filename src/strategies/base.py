from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import DocumentProfile, ExtractedDocument


class ExtractionStrategy(ABC):
    name: str

    @abstractmethod
    def extract(self, document_path: str, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float]:
        """Return extracted document, confidence score, and estimated cost in USD."""
