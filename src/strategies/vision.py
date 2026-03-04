from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader

from src.agents.errors import BudgetExceededError
from src.config import VisionConfig, VlmBudgetConfig
from src.models import BoundingBox, DocumentProfile, ExtractedDocument, FigureObject, TextBlock
from src.strategies.fast_text import FastTextExtractor


class VisionExtractor(FastTextExtractor):
    name = "vision_augmented"

    def __init__(self, thresholds: dict[str, float], vlm_budget: dict | None = None, vision_cfg: dict | None = None) -> None:
        super().__init__(thresholds=thresholds)
        budget = VlmBudgetConfig.model_validate(vlm_budget or {}).model_dump(mode="python")
        self.vision_cfg = VisionConfig.model_validate(vision_cfg or {}).model_dump(mode="python")
        self.vlm_enabled = bool(budget["enabled"])
        self.max_pages = int(budget["max_pages_per_document"])
        self.cost_per_page_usd = float(budget["cost_per_page_usd"])
        self.max_total_cost_usd = float(budget["max_total_cost_usd"])

    def extract(self, document_path: str, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float]:
        base, _, _ = super().extract(document_path, profile)
        if base.text_blocks:
            conf = max(base.confidence_score, float(self.vision_cfg["confidence_if_text_present_min"]))
            extracted = ExtractedDocument(
                doc_id=base.doc_id,
                document_name=Path(document_path).name,
                strategy_used=self.name,
                confidence_score=conf,
                text_blocks=base.text_blocks,
                tables=base.tables,
                figures=base.figures,
            )
            return extracted, conf, 0.0

        if not self.vlm_enabled:
            raise BudgetExceededError("VLM extraction is disabled by configuration (`extraction.vlm_budget.enabled=false`).")

        reader = PdfReader(document_path)
        total_pages = len(reader.pages)
        if total_pages > self.max_pages:
            raise BudgetExceededError(
                f"VLM page cap exceeded: document has {total_pages} pages "
                f"but configured cap is {self.max_pages}."
            )

        estimated_cost = total_pages * self.cost_per_page_usd
        if estimated_cost > self.max_total_cost_usd:
            raise BudgetExceededError(
                f"VLM hard cap exceeded: estimated cost ${estimated_cost:.4f} "
                f"> configured VLM cap ${self.max_total_cost_usd:.4f}."
            )

        ocr_blocks = self._ocr_pages_with_vision(reader)
        if not ocr_blocks:
            ocr_blocks = [
                TextBlock(
                    content="[vision-placeholder] OCR text would be produced by a multimodal endpoint here.",
                    page_number=1,
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=612.0, y1=792.0),
                    section_hint="vision extraction",
                    reading_order=1,
                )
            ]

        conf = float(self.vision_cfg["confidence_if_ocr_only"])
        figure_bbox_height = float(self.vision_cfg["figure_bbox_height"])
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
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=612.0, y1=figure_bbox_height),
                    caption="Vision analysis region",
                )
            ],
        )
        return extracted, conf, estimated_cost

    def _ocr_pages_with_vision(self, reader: PdfReader) -> List[TextBlock]:
        blocks: List[TextBlock] = []
        for i, page in enumerate(reader.pages, start=1):
            width = float(page.mediabox.width or 612)
            height = float(page.mediabox.height or 792)
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
