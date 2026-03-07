from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.agents.errors import BudgetExceededError
from src.models import CostTier, DocumentProfile, ExtractedDocument
from src.strategies.base import ExtractionStrategy


@dataclass
class OrchestrationResult:
    extracted: ExtractedDocument | None
    confidence: float
    total_cost: float
    per_strategy_cost: dict[str, float]
    routing_trace: list[str]
    escalated_from: str | None
    error_message: str | None
    human_review_required: bool
    final_strategy_name: str


class EscalationOrchestrator:
    """
    Explicit confidence-gated strategy orchestrator.

    Responsibilities:
    - Read `DocumentProfile` and choose initial strategy.
    - Resolve strategy chain from `EscalationConfig.chains`.
    - Execute A->B->C style escalation until confidence target is met.
    """

    def __init__(
        self,
        min_confidence: float,
        escalation_cfg: dict,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.escalation_cfg = escalation_cfg
        self.continue_on_strategy_error = bool(escalation_cfg["continue_on_strategy_error"])
        self.require_human_review_on_low_confidence = bool(escalation_cfg["require_human_review_on_low_confidence"])
        self.initial_strategy_mode = str(escalation_cfg.get("initial_strategy_mode", "profile")).strip().lower()

    def select_initial_strategy_name(self, profile: DocumentProfile) -> str:
        # Table-heavy documents should start with layout-aware extraction for structured fidelity.
        layout = getattr(profile.layout_complexity, "value", profile.layout_complexity)
        if str(layout) == "table_heavy":
            return "layout_aware"
        if self.initial_strategy_mode == "always_fast_text":
            return "fast_text"
        if profile.estimated_extraction_cost == CostTier.FAST_TEXT_SUFFICIENT:
            return "fast_text"
        if profile.estimated_extraction_cost == CostTier.NEEDS_LAYOUT_MODEL:
            return "layout_aware"
        return "vision_augmented"

    def build_chain(self, initial_strategy: str, strategies: dict[str, ExtractionStrategy]) -> list[tuple[str, ExtractionStrategy]]:
        raw_chains = self.escalation_cfg.get("chains", {})
        names = raw_chains.get(initial_strategy) if isinstance(raw_chains, dict) else None
        if not names:
            names = [initial_strategy]
        chain: list[tuple[str, ExtractionStrategy]] = []
        for n in names:
            strat = strategies.get(n)
            if strat is not None:
                chain.append((n, strat))
        if not chain and initial_strategy in strategies:
            chain.append((initial_strategy, strategies[initial_strategy]))
        return chain

    def execute(
        self,
        document_path: str,
        profile: DocumentProfile,
        strategies: dict[str, ExtractionStrategy],
        preflight_check: Callable[[str, float, float], None] | None = None,
        post_attempt_check: Callable[[str, float, float], None] | None = None,
    ) -> OrchestrationResult:
        initial = self.select_initial_strategy_name(profile)
        chain = self.build_chain(initial, strategies)

        total_cost = 0.0
        per_strategy_cost: dict[str, float] = {}
        routing_trace: list[str] = []
        escalated_from: str | None = None
        confidence = 0.0
        error_message: str | None = None
        human_review_required = False
        extracted: ExtractedDocument | None = None
        final_strategy_name = initial
        first_name: str | None = None

        for name, extractor in chain:
            if first_name is None:
                first_name = name

            if preflight_check is not None:
                preflight_check(name, total_cost, per_strategy_cost.get(name, 0.0))

            try:
                current_extracted, current_conf, cost = extractor.extract(document_path, profile)
            except BudgetExceededError as exc:
                error_message = str(exc)
                human_review_required = True
                raise
            except Exception as exc:  # noqa: BLE001
                error_message = str(exc)
                human_review_required = True
                if self.continue_on_strategy_error:
                    continue
                raise

            final_strategy_name = current_extracted.strategy_used
            total_cost += cost
            per_strategy_cost[name] = per_strategy_cost.get(name, 0.0) + cost
            if post_attempt_check is not None:
                post_attempt_check(name, total_cost, per_strategy_cost[name])
            routing_trace.append(name)

            extracted = current_extracted
            confidence = current_conf

            if name != first_name and escalated_from is None:
                escalated_from = first_name

            if confidence >= self.min_confidence:
                break

        if extracted is None or (self.require_human_review_on_low_confidence and confidence < self.min_confidence):
            human_review_required = True

        return OrchestrationResult(
            extracted=extracted,
            confidence=confidence,
            total_cost=total_cost,
            per_strategy_cost=per_strategy_cost,
            routing_trace=routing_trace,
            escalated_from=escalated_from,
            error_message=error_message,
            human_review_required=human_review_required,
            final_strategy_name=final_strategy_name,
        )
