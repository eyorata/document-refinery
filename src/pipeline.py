from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from src.agents.chunker import ChunkingEngine
from src.agents.extractor import ExtractionRouter
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryInterfaceAgent
from src.agents.triage import TriageAgent
from src.config import load_config
from src.storage import FactTableStore, build_vector_store


class RefineryPipeline:
    def __init__(self, config_path: str = "rubric/extraction_rules.yaml", output_dir: str = ".refinery") -> None:
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.triage = TriageAgent(
            domain_keywords=self.config["triage"]["domain_keywords"],
            thresholds=self.config["triage"]["thresholds"],
        )
        self.router = ExtractionRouter(config=self.config, output_dir=str(self.output_dir))
        self.chunker = ChunkingEngine(
            max_tokens=int(self.config["chunking"]["max_tokens"]),
            enabled_rules=list(self.config["chunking"].get("rules", [])),
        )
        self.indexer = PageIndexBuilder(pageindex_cfg=self.config.get("pageindex", {}))
        self.vector_store = build_vector_store(self.config.get("storage", {}))
        self.fact_table = FactTableStore(str(self.output_dir / "facts.db"))
        self.query_agent = QueryInterfaceAgent(
            self.vector_store,
            self.fact_table,
            router_cfg=self.config.get("query_agent", {}).get("router", {}),
        )

    def run(self, document_path: str, question: str = "Summarize key points"):
        processed = self.process_document(document_path)
        return self.answer_question(question, processed["chunks"], processed["page_index"])

    def process_document(self, document_path: str) -> dict[str, Any]:
        # Ensure extraction/retrieval state is scoped to the currently processed document.
        self.vector_store = build_vector_store(self.config.get("storage", {}))
        self.fact_table.clear()
        self.query_agent = QueryInterfaceAgent(
            self.vector_store,
            self.fact_table,
            router_cfg=self.config.get("query_agent", {}).get("router", {}),
        )

        profile = self.triage.profile(document_path)
        self._save_profile(profile)

        extracted = self.router.route(document_path, profile)
        self._save_extracted(profile.doc_id, extracted)
        chunks = self.chunker.chunk(extracted)
        self._save_chunks(profile.doc_id, chunks)

        self.vector_store.ingest(chunks)
        self.fact_table.ingest(chunks)

        page_index = self.indexer.build(chunks)
        self._save_pageindex(profile.doc_id, page_index)
        precision = self.indexer.evaluate_retrieval_precision(
            topic="capital expenditure projections q3",
            chunks=chunks,
            search_fn=self.vector_store.search,
            page_index=page_index,
            top_k=3,
        )
        self._save_pageindex_metrics(profile.doc_id, precision)
        return {"profile": profile, "extracted": extracted, "chunks": chunks, "page_index": page_index, "precision": precision}

    def answer_question(self, question: str, chunks, page_index):
        # Rebuild retrieval state from this document's chunks for isolated multi-turn Q&A sessions.
        self.vector_store = build_vector_store(self.config.get("storage", {}))
        self.fact_table.clear()
        self.vector_store.ingest(chunks)
        self.fact_table.ingest(chunks)
        self.query_agent = QueryInterfaceAgent(
            self.vector_store,
            self.fact_table,
            router_cfg=self.config.get("query_agent", {}).get("router", {}),
        )
        return self.query_agent.answer(question, page_index)

    def _save_profile(self, profile) -> None:
        p = self.output_dir / "profiles" / f"{profile.doc_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    def _save_extracted(self, doc_id: str, extracted) -> None:
        p = self.output_dir / "extracted" / f"{doc_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(extracted.model_dump_json(indent=2), encoding="utf-8")

    def _save_chunks(self, doc_id: str, chunks) -> None:
        p = self.output_dir / "chunks" / f"{doc_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = [c.model_dump(mode="python") for c in chunks]
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_pageindex(self, doc_id: str, index) -> None:
        p = self.output_dir / "pageindex" / f"{doc_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(index.model_dump_json(indent=2), encoding="utf-8")

    def _save_pageindex_metrics(self, doc_id: str, metrics: dict[str, float]) -> None:
        p = self.output_dir / "pageindex_metrics" / f"{doc_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
