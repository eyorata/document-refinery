from __future__ import annotations

import time
from pathlib import Path

from src.agents.errors import BudgetExceededError, HumanReviewRequiredError
from src.agents.orchestrator import EscalationOrchestrator
from src.config import ExtractionConfig
from src.models import DocumentProfile, ExtractionLedgerEntry, ExtractedDocument
from src.strategies import FastTextExtractor, LayoutExtractor, VisionExtractor
from src.strategies.base import ExtractionStrategy
from src.utils.jsonl import append_jsonl


class ExtractionRouter:
    def __init__(self, config: dict, output_dir: str) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        thresholds = config["triage"]["thresholds"]
        extraction_cfg = ExtractionConfig.model_validate(config["extraction"]).model_dump(mode="python")
        self.fast = FastTextExtractor(thresholds=thresholds, fast_cfg=extraction_cfg.get("fast_text"))
        self.layout = LayoutExtractor(thresholds=thresholds, layout_cfg=extraction_cfg.get("layout"))
        self.vision = VisionExtractor(
            thresholds=thresholds,
            vlm_budget=extraction_cfg.get("vlm_budget"),
            vision_cfg=extraction_cfg.get("vision"),
        )
        self.min_confidence = float(config["extraction"]["confidence_minimum"])
        self.enforce_hard_caps = bool(config["extraction"].get("enforce_hard_caps", True))
        self.doc_budget_usd = float(config["extraction"]["budget_per_document_usd"])
        strategy_budgets_cfg = extraction_cfg.get("strategy_budgets_usd", {})
        strategy_estimates_cfg = extraction_cfg.get("strategy_estimated_costs_usd", {})
        self.strategy_budgets: dict[str, float] = {
            "fast_text": float(strategy_budgets_cfg.get("fast_text", self.doc_budget_usd)),
            "layout_aware": float(strategy_budgets_cfg.get("layout_aware", self.doc_budget_usd)),
            "vision_augmented": float(strategy_budgets_cfg.get("vision_augmented", self.doc_budget_usd)),
        }
        self.strategy_estimated_costs: dict[str, float] = {
            "fast_text": float(strategy_estimates_cfg["fast_text"]),
            "layout_aware": float(strategy_estimates_cfg["layout_aware"]),
            "vision_augmented": float(strategy_estimates_cfg["vision_augmented"]),
        }
        self.vlm_budget = extraction_cfg["vlm_budget"]
        self.escalation_cfg = extraction_cfg["escalation"]
        self.orchestrator = EscalationOrchestrator(
            min_confidence=self.min_confidence,
            escalation_cfg=self.escalation_cfg,
        )

    def route(self, document_path: str, profile: DocumentProfile) -> ExtractedDocument:
        start = time.time()
        initial_name = self.orchestrator.select_initial_strategy_name(profile)
        strategy_map: dict[str, ExtractionStrategy] = {
            "fast_text": self.fast,
            "layout_aware": self.layout,
            "vision_augmented": self.vision,
        }
        conf = 0.0
        total_cost = 0.0
        extracted: ExtractedDocument | None = None
        routing_trace: list[str] = []
        escalated_from: str | None = None
        error_message: str | None = None
        human_review_required = False
        final_strategy_name = initial_name

        try:
            preflight_cb = (
                (lambda name, total_cost_now, strategy_total_now: self._preflight_budget_check(
                    name,
                    total_cost_now,
                    strategy_total_now,
                    profile,
                ))
                if self.enforce_hard_caps
                else None
            )
            result = self.orchestrator.execute(
                document_path=document_path,
                profile=profile,
                strategies=strategy_map,
                preflight_check=preflight_cb,
                post_attempt_check=self._post_attempt_budget_check,
            )
            extracted = result.extracted
            conf = result.confidence
            total_cost = result.total_cost
            routing_trace = result.routing_trace
            escalated_from = result.escalated_from
            error_message = result.error_message
            human_review_required = result.human_review_required
            final_strategy_name = result.final_strategy_name
            if escalated_from is None and final_strategy_name != initial_name:
                escalated_from = initial_name
            if escalated_from is None and len(routing_trace) > 1:
                escalated_from = routing_trace[0]

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
            token_usage = None
            provider = None
            if final_strategy_name == "vision_augmented":
                token_usage = getattr(self.vision, "last_token_usage", None)
                provider = getattr(self.vision, "last_provider", None)
            elif final_strategy_name == "layout_aware":
                provider = getattr(self.layout, "last_adapter_used", None)
            # Ensure we always record something in the ledger, even if extraction failed.
            entry = ExtractionLedgerEntry(
                doc_id=profile.doc_id,
                document_name=profile.document_name,
                strategy_used=final_strategy_name,
                initial_strategy=initial_name,
                confidence_score=conf,
                cost_estimate_usd=total_cost,
                processing_time_sec=elapsed,
                escalated_from=escalated_from,
                routing_trace=routing_trace,
                error_message=error_message,
                human_review_required=human_review_required,
                token_usage=token_usage,
                provider=provider,
            )
            append_jsonl(self.output_dir / "extraction_ledger.jsonl", entry.model_dump())

        if human_review_required:
            raise HumanReviewRequiredError(
                f"Extraction for {profile.document_name} completed with low confidence "
                f"({conf:.3f}); human review is required."
            )

        # Attach routing metadata to the extracted document for downstream consumers.
        extracted.routing_trace = routing_trace
        extracted.total_cost_estimate_usd = total_cost

        return extracted

    def _preflight_budget_check(
        self,
        strategy_name: str,
        total_cost: float,
        strategy_total: float,
        profile: DocumentProfile,
    ) -> None:
        projected = self._projected_strategy_cost(strategy_name, profile=profile)
        self._enforce_budget(
            total_cost + projected,
            strategy_name,
            strategy_total + projected,
            preflight=True,
        )

    def _post_attempt_budget_check(self, strategy_name: str, total_cost: float, strategy_total: float) -> None:
        self._enforce_budget(total_cost, strategy_name, strategy_total)

    def _enforce_budget(self, total_cost: float, strategy_name: str, strategy_total: float, preflight: bool = False) -> None:
        phase = "preflight " if preflight else ""
        if total_cost > self.doc_budget_usd:
            raise BudgetExceededError(
                f"Budget guard triggered ({phase}document): estimated total cost ${total_cost:.4f} "
                f"> document cap ${self.doc_budget_usd:.4f}"
            )
        cap = self.strategy_budgets.get(strategy_name)
        if cap is not None and strategy_total > cap:
            raise BudgetExceededError(
                f"Budget guard triggered ({phase}{strategy_name}): "
                f"estimated cost ${strategy_total:.4f} > strategy cap ${cap:.4f}"
            )

    def _projected_strategy_cost(self, strategy_name: str, profile: DocumentProfile) -> float:
        if strategy_name == "vision_augmented":
            cost_per_page = float(self.vlm_budget["cost_per_page_usd"])
            max_pages = int(self.vlm_budget["max_pages_per_document"])
            max_total = float(self.vlm_budget["max_total_cost_usd"])
            page_budget = max(1, min(profile.page_count, max_pages))
            return min(page_budget * cost_per_page, max_total)
        return float(self.strategy_estimated_costs.get(strategy_name, 0.0))
