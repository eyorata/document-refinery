from __future__ import annotations

import math
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
        self.stop_on_budget_exceeded = bool(budget["stop_on_budget_exceeded"])
        self.allow_partial_processing = bool(budget["allow_partial_processing"])

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
        affordable_pages = self._affordable_pages(total_pages)
        if affordable_pages <= 0:
            raise BudgetExceededError("VLM budget cap allows 0 pages; vision processing is blocked.")

        if (total_pages > affordable_pages) and (self.stop_on_budget_exceeded or not self.allow_partial_processing):
            raise BudgetExceededError(
                f"VLM cap exceeded: document has {total_pages} pages but only {affordable_pages} "
                f"pages are allowed by budget/page caps."
            )

        pages_to_process = min(total_pages, affordable_pages)
        estimated_cost = pages_to_process * self.cost_per_page_usd
        ocr_blocks = self._ocr_pages_with_vision(reader, max_pages=pages_to_process)
        if pages_to_process < total_pages:
            ocr_blocks.append(
                TextBlock(
                    content=(
                        f"[vision-budget-stop] Processed {pages_to_process}/{total_pages} pages due to "
                        "VLM budget/page caps."
                    ),
                    page_number=max(1, pages_to_process),
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=612.0, y1=80.0),
                    section_hint="vision budget cap",
                    reading_order=pages_to_process + 1,
                )
            )
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

    def _ocr_pages_with_vision(self, reader: PdfReader, max_pages: int) -> List[TextBlock]:
        blocks: List[TextBlock] = []
        for i, page in enumerate(reader.pages, start=1):
            if i > max_pages:
                break
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

    def _affordable_pages(self, total_pages: int) -> int:
        if self.cost_per_page_usd <= 0:
            cost_cap_pages = total_pages
        else:
            cost_cap_pages = int(math.floor((self.max_total_cost_usd / self.cost_per_page_usd) + 1e-9))
        return max(0, min(total_pages, self.max_pages, cost_cap_pages))
