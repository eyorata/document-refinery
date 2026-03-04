from src.agents.query_agent import QueryInterfaceAgent
from src.models import BoundingBox, LDU, PageIndexNode, PageRef
from src.storage.fact_table import FactTableStore
from src.storage.vector_store import SimpleVectorStore


def _index() -> PageIndexNode:
    return PageIndexNode(
        title="Document",
        page_start=1,
        page_end=3,
        child_sections=[
            PageIndexNode(
                title="Summary",
                page_start=1,
                page_end=3,
                child_sections=[],
                key_entities=[],
                summary="summary",
                data_types_present=["text"],
            )
        ],
        key_entities=[],
        summary="doc",
        data_types_present=["text"],
    )


def test_provenance_chain_no_hits_has_contract_fields(tmp_path):
    agent = QueryInterfaceAgent(SimpleVectorStore(), FactTableStore(str(tmp_path / "facts.db")))
    ans = agent.answer("question", _index())
    assert ans.provenance.citations == []
    assert ans.provenance.content_hash == ""
    assert ans.provenance.bbox.x0 == 0.0
    assert ans.provenance.bbox.y0 == 0.0


def test_provenance_chain_hits_populates_top_level_fields(tmp_path):
    agent = QueryInterfaceAgent(SimpleVectorStore(), FactTableStore(str(tmp_path / "facts.db")))

    hit = LDU(
        content="Revenue increased by 10%",
        chunk_type="text",
        page_refs=[PageRef(document_name="doc.pdf", page_number=2, bbox=BoundingBox(x0=10, y0=20, x1=200, y1=300))],
        bounding_box=BoundingBox(x0=10, y0=20, x1=200, y1=300),
        token_count=4,
        content_hash="hash-a",
        parent_section="Summary",
        related_chunk_hashes=[],
        metadata={},
    )
    agent.semantic_search = lambda *_, **__: [hit]

    ans = agent.answer("revenue", _index())
    assert ans.provenance.citations
    assert ans.provenance.content_hash
    assert ans.provenance.bbox.x0 == 10
    assert ans.provenance.bbox.y0 == 20
    assert ans.provenance.bbox.x1 == 200
    assert ans.provenance.bbox.y1 == 300
