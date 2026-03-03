from __future__ import annotations

from pathlib import Path

from src.models import BoundingBox, DocumentProfile, ExtractedDocument, FigureObject, TableObject, TextBlock
from src.strategies.fast_text import FastTextExtractor


class LayoutExtractor(FastTextExtractor):
    name = "layout_aware"

    def extract(self, document_path: str, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float]:
        base, base_conf, _ = super().extract(document_path, profile)
        promoted_tables = self._promote_keyword_tables(base.text_blocks)
        all_tables = base.tables + promoted_tables
        conf = max(base_conf, 0.72 if all_tables else base_conf)
        extracted = ExtractedDocument(
            doc_id=base.doc_id,
            document_name=Path(document_path).name,
            strategy_used=self.name,
            confidence_score=conf,
            text_blocks=base.text_blocks,
            tables=all_tables,
            figures=base.figures,
        )
        return extracted, conf, 0.02

    def _promote_keyword_tables(self, blocks: list[TextBlock]) -> list[TableObject]:
        out: list[TableObject] = []
        for block in blocks:
            low = block.content.lower()
            if "balance sheet" in low or "income statement" in low or "table" in low:
                out.append(
                    TableObject(
                        page_number=block.page_number,
                        bbox=block.bbox,
                        headers=["label", "value"],
                        rows=[["snippet", block.content[:200]]],
                        title="Heuristic layout table",
                    )
                )
        return out
