from __future__ import annotations

import time
from pathlib import Path

from src.agents.errors import BudgetExceededError, HumanReviewRequiredError
from src.config import ExtractionConfig
from src.models import CostTier, DocumentProfile, ExtractionLedgerEntry, ExtractedDocument
from src.strategies import FastTextExtractor, LayoutExtractor, VisionExtractor
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
        self.continue_on_strategy_error = bool(self.escalation_cfg["continue_on_strategy_error"])
        self.require_human_review_on_low_confidence = bool(self.escalation_cfg["require_human_review_on_low_confidence"])

    def route(self, document_path: str, profile: DocumentProfile) -> ExtractedDocument:
        start = time.time()
        chosen = self._select_initial(profile)

        total_cost = 0.0
        per_strategy_cost: dict[str, float] = {}
        routing_trace: list[str] = []
        escalated_from: str | None = None
        conf = 0.0
        error_message: str | None = None
        human_review_required = False
        extracted: ExtractedDocument | None = None
        final_strategy_name = chosen.name

        try:
            chain = self._build_chain(chosen.name)

            first_name: str | None = None

            for name, extractor in chain:
                if first_name is None:
                    first_name = name

                if self.enforce_hard_caps:
                    projected = self._projected_strategy_cost(name, profile)
                    self._enforce_budget(
                        total_cost + projected,
                        name,
                        per_strategy_cost.get(name, 0.0) + projected,
                        preflight=True,
                    )

                try:
                    current_extracted, current_conf, cost = extractor.extract(document_path, profile)
                except BudgetExceededError as exc:
                    error_message = str(exc)
                    human_review_required = True
                    raise
                except Exception as exc:  # noqa: BLE001
                    # Log error and attempt next strategy in the chain.
                    error_message = str(exc)
                    human_review_required = True
                    if self.continue_on_strategy_error:
                        continue
                    raise

                final_strategy_name = current_extracted.strategy_used
                total_cost += cost
                per_strategy_cost[name] = per_strategy_cost.get(name, 0.0) + cost
                self._enforce_budget(total_cost, name, per_strategy_cost[name])
                routing_trace.append(name)

                extracted = current_extracted
                conf = current_conf

                # Record if we escalated beyond the first strategy.
                if name != first_name and escalated_from is None:
                    escalated_from = first_name

                # Stop if we reach acceptable confidence.
                if conf >= self.min_confidence:
                    break

            # If no strategy produced a result, or final confidence is still low,
            # require human review.
            if extracted is None or (self.require_human_review_on_low_confidence and conf < self.min_confidence):
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

        # Attach routing metadata to the extracted document for downstream consumers.
        extracted.routing_trace = routing_trace
        extracted.total_cost_estimate_usd = total_cost

        return extracted

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

    def _build_chain(self, initial_strategy: str) -> list[tuple[str, object]]:
        strategy_map = {
            "fast_text": self.fast,
            "layout_aware": self.layout,
            "vision_augmented": self.vision,
        }
        raw_chains = self.escalation_cfg.get("chains", {})
        names = raw_chains.get(initial_strategy) if isinstance(raw_chains, dict) else None
        if not names:
            names = [initial_strategy]
        chain: list[tuple[str, object]] = []
        for n in names:
            if n in strategy_map:
                chain.append((n, strategy_map[n]))
        if not chain:
            chain.append((initial_strategy, strategy_map[initial_strategy]))
        return chain

    def _select_initial(self, profile: DocumentProfile):
        if profile.estimated_extraction_cost == CostTier.FAST_TEXT_SUFFICIENT:
            return self.fast
        if profile.estimated_extraction_cost == CostTier.NEEDS_LAYOUT_MODEL:
            return self.layout
        return self.vision
