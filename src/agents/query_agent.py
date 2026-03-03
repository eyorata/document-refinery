from __future__ import annotations

from src.models import LDU, PageIndexNode, ProvenanceChain, ProvenanceItem, QueryAnswer
from src.storage import FactTableStore, SimpleVectorStore


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

    def answer(self, question: str, page_index: PageIndexNode) -> QueryAnswer:
        nav = self.pageindex_navigate(page_index, question, k=3)
        pages = set()
        for n in nav:
            pages.update(range(n.page_start, n.page_end + 1))
        hits = self.semantic_search(question, top_k=3, allowed_pages=pages if pages else None)
        if not hits:
            return QueryAnswer(answer="No verifiable answer found.", provenance=ProvenanceChain(citations=[]))

        synthesis = " ".join(h.content[:220] for h in hits)
        provenance = ProvenanceChain(
            citations=[
                ProvenanceItem(
                    document_name=h.page_refs[0].document_name,
                    page_number=h.page_refs[0].page_number,
                    bbox=h.page_refs[0].bbox,
                    content_hash=h.content_hash,
                )
                for h in hits
            ]
        )
        return QueryAnswer(answer=synthesis[:600], provenance=provenance)

    def audit_claim(self, claim: str, page_index: PageIndexNode) -> dict[str, object]:
        ans = self.answer(claim, page_index)
        status = "verified" if ans.provenance.citations else "unverifiable"
        return {"claim": claim, "status": status, "answer": ans.answer, "provenance": ans.provenance.model_dump()}
