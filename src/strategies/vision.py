from __future__ import annotations

from pathlib import Path

from src.models import BoundingBox, DocumentProfile, ExtractedDocument, FigureObject, TableObject, TextBlock
from src.strategies.fast_text import FastTextExtractor


class VisionExtractor(FastTextExtractor):
    name = "vision_augmented"

    def extract(self, document_path: str, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float]:
        base, _, _ = super().extract(document_path, profile)
        if not base.text_blocks:
            base.text_blocks.append(
                TextBlock(
                    content="[vision-placeholder] OCR text would be produced by multimodal endpoint.",
                    page_number=1,
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=612.0, y1=792.0),
                    section_hint="vision extraction",
                    reading_order=1,
                )
            )
        conf = 0.88
        extracted = ExtractedDocument(
            doc_id=base.doc_id,
            document_name=Path(document_path).name,
            strategy_used=self.name,
            confidence_score=conf,
            text_blocks=base.text_blocks,
            tables=base.tables,
            figures=base.figures + [
                FigureObject(page_number=1, bbox=BoundingBox(x0=0.0, y0=0.0, x1=612.0, y1=260.0), caption="Vision analysis region")
            ],
        )
        return extracted, conf, 0.10
