from __future__ import annotations

import time
from pathlib import Path

from src.models import CostTier, DocumentProfile, ExtractionLedgerEntry, ExtractedDocument
from src.strategies import FastTextExtractor, LayoutExtractor, VisionExtractor
from src.utils.jsonl import append_jsonl


class BudgetExceededError(RuntimeError):
    pass


class HumanReviewRequiredError(RuntimeError):
    pass


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

    def route(self, document_path: str, profile: DocumentProfile) -> ExtractedDocument:
        start = time.time()
        chosen = self._select_initial(profile)

        total_cost = 0.0
        escalated_from: str | None = None
        conf = 0.0
        error_message: str | None = None
        human_review_required = False
        extracted: ExtractedDocument | None = None
        final_strategy_name = chosen.name

        try:
            # Initial extraction
            extracted, conf, cost = chosen.extract(document_path, profile)
            final_strategy_name = extracted.strategy_used
            total_cost += cost
            self._enforce_budget(total_cost)

            # Escalate A -> B if needed
            if conf < self.min_confidence and chosen.name == "fast_text":
                escalated_from = chosen.name
                extracted, conf, cost = self.layout.extract(document_path, profile)
                final_strategy_name = extracted.strategy_used
                total_cost += cost
                self._enforce_budget(total_cost)

            # Escalate A/B -> C if still low confidence
            if conf < self.min_confidence and chosen.name in {"fast_text", "layout_aware"}:
                if escalated_from is None:
                    escalated_from = chosen.name
                extracted, conf, cost = self.vision.extract(document_path, profile)
                final_strategy_name = extracted.strategy_used
                total_cost += cost
                self._enforce_budget(total_cost)

            # If even the final strategy is low confidence, mark for human review.
            if conf < self.min_confidence:
                human_review_required = True

        except BudgetExceededError as exc:
            error_message = str(exc)
            human_review_required = True
            raise
        except Exception as exc:  # noqa: BLE001 - we want to capture and log all errors here
            error_message = str(exc)
            human_review_required = True
            raise
        finally:
            elapsed = time.time() - start
            # Ensure we always record something in the ledger, even if extraction failed.
            entry = ExtractionLedgerEntry(
                doc_id=profile.doc_id,
                document_name=Path(document_path).name,
                strategy_used=final_strategy_name,
                confidence_score=conf,
                cost_estimate_usd=total_cost,
                processing_time_sec=elapsed,
                escalated_from=escalated_from,
                error_message=error_message,
                human_review_required=human_review_required,
            )
            append_jsonl(self.output_dir / "extraction_ledger.jsonl", entry.model_dump())

        if human_review_required:
            raise HumanReviewRequiredError(
                f"Extraction for {profile.document_name} completed with low confidence "
                f"({conf:.3f}); human review is required."
            )

        return extracted

    def _select_initial(self, profile: DocumentProfile):
        if profile.estimated_extraction_cost == CostTier.FAST_TEXT_SUFFICIENT:
            return self.fast
        if profile.estimated_extraction_cost == CostTier.NEEDS_LAYOUT_MODEL:
            return self.layout
        return self.vision
