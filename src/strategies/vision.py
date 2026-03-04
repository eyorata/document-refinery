from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader

from src.models import BoundingBox, DocumentProfile, ExtractedDocument, FigureObject, TextBlock
from src.strategies.fast_text import FastTextExtractor


class VisionExtractor(FastTextExtractor):
    """
    Vision-augmented extractor intended for scanned / image-heavy PDFs.

    Design notes:
    - Today this class still uses a lightweight placeholder OCR step so the
      pipeline can run end-to-end for scanned documents.
    - The structure is intentionally prepared so you can later plug in a local
      VLM / OCR model (e.g. LM Studio) inside `_ocr_pages_with_vision`.
    - Downstream agents see the same normalized `ExtractedDocument` schema as
      the fast-text and layout strategies.
    """

    name = "vision_augmented"

    def extract(self, document_path: str, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float]:
        """
        Entry point used by `ExtractionRouter`.

        Strategy:
        - Run the fast-text baseline first (re-using existing logic for any
          text stream that might still be present).
        - If pages are effectively "empty" (scanned image), synthesize
          `TextBlock`s via `_ocr_pages_with_vision`.
        - Return an `ExtractedDocument` tagged with this strategy name so
          provenance and the ledger remain accurate.
        """
        base, _, _ = super().extract(document_path, profile)

        # If we already have meaningful text blocks, keep them – this handles
        # mixed-mode documents gracefully.
        if base.text_blocks:
            conf = max(base.confidence_score, 0.9)
            extracted = ExtractedDocument(
                doc_id=base.doc_id,
                document_name=Path(document_path).name,
                strategy_used=self.name,
                confidence_score=conf,
                text_blocks=base.text_blocks,
                tables=base.tables,
                figures=base.figures,
            )
            # Very rough cost placeholder for a "vision" pass.
            return extracted, conf, 0.10

        # No usable text blocks: treat as scanned and fall back to vision/OCR.
        ocr_blocks = self._ocr_pages_with_vision(document_path)

        if not ocr_blocks:
            # Absolute fallback to preserve pipeline behavior if vision is not
            # yet wired up.
            ocr_blocks = [
                TextBlock(
                    content="[vision-placeholder] OCR text would be produced by a multimodal endpoint here.",
                    page_number=1,
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=612.0, y1=792.0),
                    section_hint="vision extraction",
                    reading_order=1,
                )
            ]

        conf = 0.88
        extracted = ExtractedDocument(
            doc_id=base.doc_id,
            document_name=Path(document_path).name,
            strategy_used=self.name,
            confidence_score=conf,
            text_blocks=ocr_blocks,
            tables=base.tables,
            figures=base.figures
            + [
                FigureObject(
                    page_number=1,
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=612.0, y1=260.0),
                    caption="Vision analysis region",
                )
            ],
        )
        return extracted, conf, 0.10

    def _ocr_pages_with_vision(self, document_path: str) -> List[TextBlock]:
        """
        Placeholder hook for OCR / vision model integration.

        This is where you should call your local LM Studio VLM once you're
        ready:
        - Render each PDF page to an image (e.g. with PyMuPDF / pdf2image).
        - Send images + an extraction prompt to the LM Studio HTTP endpoint.
        - Parse the response into per-page strings.
        - Convert those strings into `TextBlock` instances with bounding boxes.

        For now this method only produces very lightweight, page-level
        placeholders so that the rest of the pipeline can be validated.
        """
        reader = PdfReader(document_path)
        blocks: List[TextBlock] = []

        for i, page in enumerate(reader.pages, start=1):
            width = float(page.mediabox.width or 612)
            height = float(page.mediabox.height or 792)

            # TODO: replace this with real OCR using a VLM via LM Studio.
            blocks.append(
                TextBlock(
                    content=f"[vision-placeholder] Page {i}: OCR text would be inserted here from the vision model.",
                    page_number=i,
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=width, y1=height),
                    section_hint="vision extraction",
                    reading_order=i,
                )
            )

        return blocks
