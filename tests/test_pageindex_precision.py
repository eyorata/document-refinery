from src.agents.indexer import PageIndexBuilder
from src.models import BoundingBox, LDU, PageRef


def _chunk(content: str, page: int, section: str) -> LDU:
    return LDU(
        content=content,
        chunk_type="text",
        page_refs=[PageRef(document_name="sample.pdf", page_number=page, bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10))],
        bounding_box=BoundingBox(x0=0, y0=0, x1=10, y1=10),
        parent_section=section,
        token_count=max(1, len(content.split())),
        content_hash=f"h-{page}-{section}",
        related_chunk_hashes=[],
        metadata={},
    )


def test_pageindex_precision_eval_returns_with_and_without_metrics():
    chunks = [
        _chunk("Revenue increased in Q3", 1, "Financial Summary"),
        _chunk("Capital expenditure projections are listed", 2, "Capex"),
        _chunk("Random appendix notes", 10, "Appendix"),
    ]
    builder = PageIndexBuilder()
    tree = builder.build(chunks)

    def search_fn(query: str, top_k: int, filter_pages: set[int] | None):
        items = chunks
        if filter_pages is not None:
            items = [c for c in items if c.page_refs[0].page_number in filter_pages]
        return items[:top_k]

    metrics = builder.evaluate_retrieval_precision(
        topic="capital expenditure projections",
        chunks=chunks,
        search_fn=search_fn,
        page_index=tree,
        top_k=2,
    )
    assert "with_pageindex" in metrics
    assert "without_pageindex" in metrics
    assert 0.0 <= metrics["with_pageindex"] <= 1.0
    assert 0.0 <= metrics["without_pageindex"] <= 1.0
