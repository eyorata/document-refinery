from __future__ import annotations

import time
from pathlib import Path

from src.models import CostTier, DocumentProfile, ExtractionLedgerEntry
from src.strategies import FastTextExtractor, LayoutExtractor, VisionExtractor
from src.utils.jsonl import append_jsonl


class ExtractionRouter:
    def __init__(self, config: dict, output_dir: str) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        thresholds = config["triage"]["thresholds"]
        self.fast = FastTextExtractor(thresholds=thresholds)
        self.layout = LayoutExtractor(thresholds=thresholds)
        self.vision = VisionExtractor(thresholds=thresholds)
        self.min_confidence = float(config["extraction"]["confidence_minimum"])
        self.doc_budget_usd = float(config["extraction"]["budget_per_document_usd"])

    def route(self, document_path: str, profile: DocumentProfile):
        start = time.time()
        chosen = self._select_initial(profile)
        extracted, conf, cost = chosen.extract(document_path, profile)
        escalated_from = None

        if conf < self.min_confidence and chosen.name == "fast_text":
            escalated_from = chosen.name
            extracted, conf, cost = self.layout.extract(document_path, profile)

        if conf < self.min_confidence and chosen.name in {"fast_text", "layout_aware"}:
            escalated_from = chosen.name if not escalated_from else escalated_from
            extracted, conf, cost = self.vision.extract(document_path, profile)

        if cost > self.doc_budget_usd:
            raise RuntimeError(f"Budget guard triggered: estimated cost ${cost:.4f} > ${self.doc_budget_usd:.4f}")

        elapsed = time.time() - start
        entry = ExtractionLedgerEntry(
            doc_id=profile.doc_id,
            document_name=Path(document_path).name,
            strategy_used=extracted.strategy_used,
            confidence_score=conf,
            cost_estimate_usd=cost,
            processing_time_sec=elapsed,
            escalated_from=escalated_from,
        )
        append_jsonl(self.output_dir / "extraction_ledger.jsonl", entry.model_dump())
        return extracted

    def _select_initial(self, profile: DocumentProfile):
        if profile.estimated_extraction_cost == CostTier.FAST_TEXT_SUFFICIENT:
            return self.fast
        if profile.estimated_extraction_cost == CostTier.NEEDS_LAYOUT_MODEL:
            return self.layout
        return self.vision
