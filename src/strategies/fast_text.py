from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from src.models import BoundingBox, DocumentProfile, ExtractedDocument, FigureObject, TableObject, TextBlock
from src.strategies.base import ExtractionStrategy


class FastTextExtractor(ExtractionStrategy):
    name = "fast_text"

    def __init__(self, thresholds: dict[str, float]) -> None:
        self.thresholds = thresholds

    def extract(self, document_path: str, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float]:
        reader = PdfReader(document_path)
        text_blocks: list[TextBlock] = []
        tables: list[TableObject] = []
        figures: list[FigureObject] = []

        char_counts: list[int] = []
        densities: list[float] = []
        image_ratios: list[float] = []

        for i, page in enumerate(reader.pages, start=1):
            txt = page.extract_text() or ""
            width = float(page.mediabox.width or 612)
            height = float(page.mediabox.height or 792)
            page_area = max(width * height, 1.0)
            char_count = len(txt)
            images = len(getattr(page, "images", []))
            image_ratio = min(images / max(self.thresholds.get("max_images_for_ratio", 10), 1), 1.0)

            char_counts.append(char_count)
            densities.append(char_count / page_area)
            image_ratios.append(image_ratio)

            if txt.strip():
                text_blocks.append(
                    TextBlock(
                        content=txt.strip(),
                        page_number=i,
                        bbox=BoundingBox(x0=0.0, y0=0.0, x1=width, y1=height),
                        section_hint=self._section_hint(txt),
                        reading_order=i,
                    )
                )
                tables.extend(self._detect_pipe_tables(txt, i, width, height))

        conf = self.score_confidence(char_counts, densities, image_ratios)
        extracted = ExtractedDocument(
            doc_id=profile.doc_id,
            document_name=Path(document_path).name,
            strategy_used=self.name,
            confidence_score=conf,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
        )
        return extracted, conf, 0.002

    def score_confidence(self, char_counts: list[int], densities: list[float], image_ratios: list[float]) -> float:
        if not char_counts:
            return 0.0
        avg_chars = sum(char_counts) / len(char_counts)
        avg_density = sum(densities) / len(densities) if densities else 0.0
        avg_img = sum(image_ratios) / len(image_ratios) if image_ratios else 0.0

        char_score = min(avg_chars / self.thresholds.get("target_chars_per_page", 350), 1.0)
        density_score = min(avg_density / self.thresholds.get("target_density", 0.001), 1.0)
        image_penalty = max(0.0, 1.0 - avg_img)

        return max(0.0, min(1.0, (char_score * 0.45) + (density_score * 0.35) + (image_penalty * 0.20)))

    def _section_hint(self, text: str) -> str | None:
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        return first[:80] if first else None

    def _detect_pipe_tables(self, text: str, page: int, width: float, height: float) -> list[TableObject]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        table_lines = [ln for ln in lines if "|" in ln and len(ln.split("|")) >= 3]
        if len(table_lines) < 2:
            return []

        headers = [c.strip() for c in table_lines[0].split("|") if c.strip()]
        rows = [[c.strip() for c in row.split("|") if c.strip()] for row in table_lines[1:]]
        rows = [r for r in rows if r]
        if not headers or not rows:
            return []

        return [
            TableObject(
                page_number=page,
                bbox=BoundingBox(x0=0.0, y0=0.0, x1=width, y1=height * 0.45),
                headers=headers,
                rows=rows,
                title="Detected pipe-delimited table",
            )
        ]
