from pathlib import Path

import pytest

from src.agents.errors import BudgetExceededError
from src.agents.extractor import ExtractionRouter
from src.models import CostTier, DocumentProfile, ExtractedDocument, LayoutComplexity, OriginType


def _profile(cost_tier: CostTier, page_count: int = 3) -> DocumentProfile:
    return DocumentProfile(
        doc_id="abc123",
        document_name="dummy.pdf",
        page_count=page_count,
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        language_code="en",
        language_confidence=0.9,
        domain_hint="general",
        estimated_extraction_cost=cost_tier,
        avg_char_density=0.001,
        avg_image_ratio=0.0,
        triage_confidence=0.9,
    )


def _extracted(strategy: str) -> ExtractedDocument:
    return ExtractedDocument(
        doc_id="abc123",
        document_name="dummy.pdf",
        strategy_used=strategy,
        confidence_score=0.95,
        text_blocks=[],
        tables=[],
        figures=[],
    )


def test_preflight_budget_blocks_fast_strategy(tmp_path: Path):
    cfg = {
        "triage": {"thresholds": {}},
        "extraction": {
            "confidence_minimum": 0.65,
            "budget_per_document_usd": 0.05,
            "enforce_hard_caps": True,
            "strategy_budgets_usd": {"fast_text": 0.05, "layout_aware": 0.15, "vision_augmented": 0.15},
            "strategy_estimated_costs_usd": {"fast_text": 0.10, "layout_aware": 0.02, "vision_augmented": 0.10},
            "vlm_budget": {"enabled": True, "max_pages_per_document": 25, "cost_per_page_usd": 0.01, "max_total_cost_usd": 0.20},
            "layout": {"estimated_cost_usd": 0.02, "adapter": {"provider": "heuristic", "options": {}}},
        },
        "chunking": {"max_tokens": 500},
    }
    router = ExtractionRouter(cfg, output_dir=str(tmp_path))
    router.fast.extract = lambda *_: (_extracted("fast_text"), 0.95, 0.001)  # pragma: no cover

    with pytest.raises(BudgetExceededError):
        router.route("dummy.pdf", _profile(CostTier.FAST_TEXT_SUFFICIENT))


def test_preflight_budget_blocks_vision_by_page_cost(tmp_path: Path):
    cfg = {
        "triage": {"thresholds": {}},
        "extraction": {
            "confidence_minimum": 0.65,
            "budget_per_document_usd": 1.00,
            "enforce_hard_caps": True,
            "strategy_budgets_usd": {"fast_text": 0.05, "layout_aware": 0.15, "vision_augmented": 0.15},
            "strategy_estimated_costs_usd": {"fast_text": 0.002, "layout_aware": 0.02, "vision_augmented": 0.10},
            "vlm_budget": {"enabled": True, "max_pages_per_document": 25, "cost_per_page_usd": 0.01, "max_total_cost_usd": 0.20},
            "layout": {"estimated_cost_usd": 0.02, "adapter": {"provider": "heuristic", "options": {}}},
        },
        "chunking": {"max_tokens": 500},
    }
    router = ExtractionRouter(cfg, output_dir=str(tmp_path))
    router.vision.extract = lambda *_: (_extracted("vision_augmented"), 0.95, 0.10)  # pragma: no cover

    with pytest.raises(BudgetExceededError):
        router.route("dummy.pdf", _profile(CostTier.NEEDS_VISION_MODEL, page_count=50))
