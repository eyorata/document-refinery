from src.models import BoundingBox, CostTier, DocumentProfile, LayoutComplexity, OriginType, TextBlock
from src.strategies.layout_aware import DoclingLayoutAdapter, build_layout_adapter


def _profile() -> DocumentProfile:
    return DocumentProfile(
        doc_id="abc123",
        document_name="dummy.pdf",
        page_count=1,
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.TABLE_HEAVY,
        language_code="en",
        language_confidence=0.9,
        domain_hint="general",
        estimated_extraction_cost=CostTier.NEEDS_LAYOUT_MODEL,
        avg_char_density=0.001,
        avg_image_ratio=0.1,
        triage_confidence=0.9,
    )


def test_build_layout_adapter_docling():
    adapter = build_layout_adapter({"provider": "docling", "options": {}})
    assert isinstance(adapter, DoclingLayoutAdapter)


def test_docling_adapter_falls_back_to_heuristic_when_docling_missing():
    adapter = build_layout_adapter({"provider": "docling", "options": {"strict": False}})
    blocks = [
        TextBlock(
            content="Balance Sheet\nAssets | Value",
            page_number=1,
            bbox=BoundingBox(x0=0.0, y0=0.0, x1=612.0, y1=792.0),
            section_hint=None,
            reading_order=1,
        )
    ]
    tables = adapter.promote_tables(blocks, "dummy.pdf", _profile())
    assert tables
    assert tables[0].title
