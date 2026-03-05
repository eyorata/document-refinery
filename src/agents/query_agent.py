from __future__ import annotations

from src.models import BoundingBox, LDU, PageIndexNode, ProvenanceChain, ProvenanceItem, QueryAnswer
from src.storage import FactTableStore, SimpleVectorStore
from src.utils.hashing import stable_hash


class QueryInterfaceAgent:
    def __init__(self, vector_store: SimpleVectorStore, fact_table: FactTableStore) -> None:
        self.vector_store = vector_store
        self.fact_table = fact_table

    def pageindex_navigate(self, index: PageIndexNode, topic: str, k: int = 3) -> list[PageIndexNode]:
        low = topic.lower()
        scored = []
        for node in index.child_sections:
            score = node.title.lower().count(low) + node.summary.lower().count(low)
            scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for s, n in scored[:k]]

    def semantic_search(self, query: str, top_k: int = 3, allowed_pages: set[int] | None = None) -> list[LDU]:
        return self.vector_store.search(query, top_k=top_k, filter_pages=allowed_pages)

    def structured_query(self, sql: str):
        return self.fact_table.query(sql)

    def tools(self) -> dict[str, object]:
        return {
            "pageindex_navigate": self.pageindex_navigate,
            "semantic_search": self.semantic_search,
            "structured_query": self.structured_query,
        }

    def build_langgraph(self):
        """
        Optional LangGraph wiring for environments where langgraph is installed.
        The runtime pipeline works without this dependency.
        """
        try:
            from langgraph.graph import END, StateGraph  # type: ignore[import-not-found]
        except Exception:
            return None

        class _State(dict):
            pass

        graph = StateGraph(_State)
        graph.add_node("pageindex_navigate", lambda s: s)
        graph.add_node("semantic_search", lambda s: s)
        graph.add_node("structured_query", lambda s: s)
        graph.set_entry_point("pageindex_navigate")
        graph.add_edge("pageindex_navigate", "semantic_search")
        graph.add_edge("semantic_search", "structured_query")
        graph.add_edge("structured_query", END)
        return graph.compile()

    def answer(self, question: str, page_index: PageIndexNode) -> QueryAnswer:
        nav = self.pageindex_navigate(page_index, question, k=3)
        pages = set()
        for n in nav:
            pages.update(range(n.page_start, n.page_end + 1))
        hits = self.semantic_search(question, top_k=3, allowed_pages=pages if pages else None)
        if not hits:
            return QueryAnswer(
                answer="No verifiable answer found.",
                provenance=ProvenanceChain(
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0),
                    content_hash="",
                    citations=[],
                ),
            )

        synthesis = " ".join(h.content[:220] for h in hits)
        citations = [
            ProvenanceItem(
                document_name=h.page_refs[0].document_name,
                page_number=h.page_refs[0].page_number,
                bbox=h.page_refs[0].bbox,
                content_hash=h.content_hash,
            )
            for h in hits
        ]
        provenance = ProvenanceChain(
            bbox=self._aggregate_bbox(citations),
            content_hash=stable_hash("|".join(c.content_hash for c in citations)),
            citations=citations,
        )
        return QueryAnswer(answer=synthesis[:600], provenance=provenance)

    def audit_claim(self, claim: str, page_index: PageIndexNode) -> dict[str, object]:
        ans = self.answer(claim, page_index)
        status = "verified" if ans.provenance.citations else "unverifiable"
        return {"claim": claim, "status": status, "answer": ans.answer, "provenance": ans.provenance.model_dump()}

    def _aggregate_bbox(self, citations: list[ProvenanceItem]) -> BoundingBox:
        if not citations:
            return BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)
        x0 = min(c.bbox.x0 for c in citations)
        y0 = min(c.bbox.y0 for c in citations)
        x1 = max(c.bbox.x1 for c in citations)
        y1 = max(c.bbox.y1 for c in citations)
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
