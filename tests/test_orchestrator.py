import pytest

from src.agents.orchestrator import EscalationOrchestrator
from src.models import BoundingBox, CostTier, DocumentProfile, ExtractedDocument, LayoutComplexity, OriginType, TextBlock
from src.strategies.base import ExtractionStrategy


class _FakeStrategy(ExtractionStrategy):
    def __init__(self, name: str, conf: float, cost: float, fail: bool = False) -> None:
        self.name = name
        self.conf = conf
        self.cost = cost
        self.fail = fail

    def extract(self, document_path: str, profile: DocumentProfile):
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return (
            ExtractedDocument(
                doc_id=profile.doc_id,
                document_name=profile.document_name,
                strategy_used=self.name,
                confidence_score=self.conf,
                text_blocks=[
                    TextBlock(
                        content=f"{self.name} output",
                        page_number=1,
                        bbox=BoundingBox(x0=0.0, y0=0.0, x1=10.0, y1=10.0),
                        section_hint=self.name,
                        reading_order=1,
                    )
                ],
                tables=[],
                figures=[],
            ),
            self.conf,
            self.cost,
        )


def _profile(cost_tier: CostTier) -> DocumentProfile:
    return DocumentProfile(
        doc_id="abc123",
        document_name="dummy.pdf",
        page_count=1,
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        language_code="en",
        language_confidence=0.9,
        domain_hint="general",
        estimated_extraction_cost=cost_tier,
        avg_char_density=0.001,
        avg_image_ratio=0.1,
        triage_confidence=0.9,
    )


def _orchestrator(continue_on_error: bool = True) -> EscalationOrchestrator:
    return EscalationOrchestrator(
        min_confidence=0.65,
        escalation_cfg={
            "continue_on_strategy_error": continue_on_error,
            "require_human_review_on_low_confidence": True,
            "chains": {
                "fast_text": ["fast_text", "layout_aware", "vision_augmented"],
                "layout_aware": ["layout_aware", "vision_augmented"],
                "vision_augmented": ["vision_augmented"],
            },
        },
    )


def test_select_initial_strategy_name():
    orch = _orchestrator()
    assert orch.select_initial_strategy_name(_profile(CostTier.FAST_TEXT_SUFFICIENT)) == "fast_text"
    assert orch.select_initial_strategy_name(_profile(CostTier.NEEDS_LAYOUT_MODEL)) == "layout_aware"
    assert orch.select_initial_strategy_name(_profile(CostTier.NEEDS_VISION_MODEL)) == "vision_augmented"


def test_confidence_gated_escalation_a_to_b_to_c():
    orch = _orchestrator()
    strategies = {
        "fast_text": _FakeStrategy("fast_text", conf=0.3, cost=0.01),
        "layout_aware": _FakeStrategy("layout_aware", conf=0.5, cost=0.02),
        "vision_augmented": _FakeStrategy("vision_augmented", conf=0.9, cost=0.03),
    }
    result = orch.execute("dummy.pdf", _profile(CostTier.FAST_TEXT_SUFFICIENT), strategies)
    assert result.routing_trace == ["fast_text", "layout_aware", "vision_augmented"]
    assert result.confidence == 0.9
    assert result.total_cost == 0.06
    assert result.human_review_required is False


def test_confidence_gate_stops_when_threshold_met():
    orch = _orchestrator()
    strategies = {
        "fast_text": _FakeStrategy("fast_text", conf=0.8, cost=0.01),
        "layout_aware": _FakeStrategy("layout_aware", conf=0.9, cost=0.02),
        "vision_augmented": _FakeStrategy("vision_augmented", conf=0.95, cost=0.03),
    }
    result = orch.execute("dummy.pdf", _profile(CostTier.FAST_TEXT_SUFFICIENT), strategies)
    assert result.routing_trace == ["fast_text"]
    assert result.final_strategy_name == "fast_text"


def test_error_behavior_respects_continue_flag():
    orch = _orchestrator(continue_on_error=False)
    strategies = {
        "fast_text": _FakeStrategy("fast_text", conf=0.1, cost=0.01, fail=True),
        "layout_aware": _FakeStrategy("layout_aware", conf=0.9, cost=0.02),
        "vision_augmented": _FakeStrategy("vision_augmented", conf=0.9, cost=0.03),
    }
    with pytest.raises(RuntimeError):
        orch.execute("dummy.pdf", _profile(CostTier.FAST_TEXT_SUFFICIENT), strategies)
