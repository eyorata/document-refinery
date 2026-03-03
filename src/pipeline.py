from __future__ import annotations

import json
from pathlib import Path

from src.agents.chunker import ChunkingEngine
from src.agents.extractor import ExtractionRouter
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryInterfaceAgent
from src.agents.triage import TriageAgent
from src.config import load_config
from src.storage import FactTableStore, SimpleVectorStore


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
        self.chunker = ChunkingEngine(max_tokens=int(self.config["chunking"]["max_tokens"]))
        self.indexer = PageIndexBuilder()
        self.vector_store = SimpleVectorStore()
        self.fact_table = FactTableStore(str(self.output_dir / "facts.db"))
        self.query_agent = QueryInterfaceAgent(self.vector_store, self.fact_table)

    def run(self, document_path: str):
        profile = self.triage.profile(document_path)
        self._save_profile(profile)

        extracted = self.router.route(document_path, profile)
        chunks = self.chunker.chunk(extracted)

        for c in chunks:
            for ref in c.page_refs:
                ref.document_name = profile.document_name

        self.vector_store.ingest(chunks)
        self.fact_table.ingest(chunks)

        page_index = self.indexer.build(chunks)
        self._save_pageindex(profile.doc_id, page_index)

        answer = self.query_agent.answer("Summarize key points", page_index)
        return answer

    def _save_profile(self, profile) -> None:
        p = self.output_dir / "profiles" / f"{profile.doc_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    def _save_pageindex(self, doc_id: str, index) -> None:
        p = self.output_dir / "pageindex" / f"{doc_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(index.model_dump_json(indent=2), encoding="utf-8")
