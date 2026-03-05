from src.agents.chunker import ChunkingEngine
from src.models import BoundingBox, ExtractedDocument, FigureObject, TableObject, TextBlock


def _doc() -> ExtractedDocument:
    return ExtractedDocument(
        doc_id="doc1",
        document_name="sample.pdf",
        strategy_used="layout_aware",
        confidence_score=0.9,
        text_blocks=[
            TextBlock(
                content="Financial Summary\nSee Table 1 and Figure 1 for details.",
                page_number=1,
                bbox=BoundingBox(x0=0, y0=0, x1=612, y1=792),
                section_hint="Financial Summary",
                reading_order=1,
            )
        ],
        tables=[
            TableObject(
                page_number=1,
                bbox=BoundingBox(x0=10, y0=100, x1=600, y1=300),
                headers=["metric", "value"],
                rows=[["revenue", "$4.2B"]],
                title="Table 1",
            )
        ],
        figures=[
            FigureObject(
                page_number=1,
                bbox=BoundingBox(x0=20, y0=320, x1=590, y1=500),
                caption="Figure 1: Revenue trend",
            )
        ],
    )


def test_chunking_enforces_table_and_figure_metadata_and_cross_refs():
    engine = ChunkingEngine(max_tokens=500)
    chunks = engine.chunk(_doc())

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    figure_chunks = [c for c in chunks if c.chunk_type == "figure"]
    text_chunks = [c for c in chunks if c.chunk_type == "text"]

    assert table_chunks and figure_chunks and text_chunks
    assert "headers" in table_chunks[0].metadata
    assert "caption" in figure_chunks[0].metadata
    assert table_chunks[0].parent_section == "Financial Summary"
    assert figure_chunks[0].parent_section == "Financial Summary"
    assert table_chunks[0].content_hash in text_chunks[0].related_chunk_hashes
    assert figure_chunks[0].content_hash in text_chunks[0].related_chunk_hashes


def test_numbered_list_only_splits_when_exceeding_max_tokens():
    content = "\n".join(f"{i}. item {i}" for i in range(1, 21))
    doc = ExtractedDocument(
        doc_id="doc2",
        document_name="sample.pdf",
        strategy_used="fast_text",
        confidence_score=0.9,
        text_blocks=[
            TextBlock(
                content=content,
                page_number=1,
                bbox=BoundingBox(x0=0, y0=0, x1=612, y1=792),
                section_hint="Checklist",
                reading_order=1,
            )
        ],
        tables=[],
        figures=[],
    )

    small_engine = ChunkingEngine(max_tokens=10)
    split_chunks = [c for c in small_engine.chunk(doc) if c.chunk_type == "list"]
    assert len(split_chunks) > 1

    large_engine = ChunkingEngine(max_tokens=200)
    single_chunks = [c for c in large_engine.chunk(doc) if c.chunk_type == "list"]
    assert len(single_chunks) == 1
