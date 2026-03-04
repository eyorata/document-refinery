from pathlib import Path

import pytest

from src.agents.errors import BudgetExceededError
from src.models import BoundingBox, CostTier, DocumentProfile, ExtractedDocument, LayoutComplexity, OriginType
from src.strategies.vision import VisionExtractor


def _profile(page_count: int) -> DocumentProfile:
    return DocumentProfile(
        doc_id="abc123",
        document_name="dummy.pdf",
        page_count=page_count,
        origin_type=OriginType.SCANNED_IMAGE,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        language_code="en",
        language_confidence=0.9,
        domain_hint="general",
        estimated_extraction_cost=CostTier.NEEDS_VISION_MODEL,
        avg_char_density=0.0001,
        avg_image_ratio=0.9,
        triage_confidence=0.9,
    )


class _FakeMediaBox:
    width = 612
    height = 792


class _FakePage:
    mediabox = _FakeMediaBox()


class _FakeReader:
    def __init__(self, num_pages: int) -> None:
        self.pages = [_FakePage() for _ in range(num_pages)]


def _empty_fast_extract(self, document_path: str, profile: DocumentProfile):
    return (
        ExtractedDocument(
            doc_id=profile.doc_id,
            document_name=Path(document_path).name,
            strategy_used="fast_text",
            confidence_score=0.1,
            text_blocks=[],
            tables=[],
            figures=[],
        ),
        0.1,
        0.0,
    )


def test_vision_hard_stop_on_budget_exceeded(monkeypatch):
    monkeypatch.setattr("src.strategies.vision.FastTextExtractor.extract", _empty_fast_extract)
    monkeypatch.setattr("src.strategies.vision.PdfReader", lambda _: _FakeReader(5))

    ex = VisionExtractor(
        thresholds={},
        vlm_budget={
            "enabled": True,
            "max_pages_per_document": 2,
            "cost_per_page_usd": 0.01,
            "max_total_cost_usd": 0.20,
            "stop_on_budget_exceeded": True,
            "allow_partial_processing": False,
        },
        vision_cfg={},
    )

    with pytest.raises(BudgetExceededError):
        ex.extract("dummy.pdf", _profile(page_count=5))


def test_vision_partial_processing_when_configured(monkeypatch):
    monkeypatch.setattr("src.strategies.vision.FastTextExtractor.extract", _empty_fast_extract)
    monkeypatch.setattr("src.strategies.vision.PdfReader", lambda _: _FakeReader(5))

    ex = VisionExtractor(
        thresholds={},
        vlm_budget={
            "enabled": True,
            "max_pages_per_document": 2,
            "cost_per_page_usd": 0.01,
            "max_total_cost_usd": 0.20,
            "stop_on_budget_exceeded": False,
            "allow_partial_processing": True,
        },
        vision_cfg={},
    )
    extracted, _, cost = ex.extract("dummy.pdf", _profile(page_count=5))
    assert cost == pytest.approx(0.02)
    assert len(extracted.text_blocks) >= 3  # 2 OCR blocks + budget-stop note
    assert any("vision-budget-stop" in b.content for b in extracted.text_blocks)
