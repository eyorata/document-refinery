# Document Intelligence Refinery

Production-oriented baseline for the Week 3 TRP1 challenge: a typed, multi-stage, strategy-routed document intelligence pipeline.

## What this implements

- 5-stage architecture:
1. `TriageAgent` (`src/agents/triage.py`)
2. `ExtractionRouter` + strategies (`src/agents/extractor.py`, `src/strategies/*`)
3. `ChunkingEngine` + `ChunkValidator` (`src/agents/chunker.py`)
4. `PageIndexBuilder` (`src/agents/indexer.py`)
5. `QueryInterfaceAgent` (`src/agents/query_agent.py`)

- Core typed models in `src/models/schemas.py`:
- `DocumentProfile`, `ExtractedDocument`, `LDU`, `PageIndexNode`, `ProvenanceChain`, and related schemas.

- Externalized routing and chunking rules:
- `rubric/extraction_rules.yaml`

- Artifacts emitted to `.refinery/`:
- `profiles/{doc_id}.json`
- `extraction_ledger.jsonl`
- `pageindex/{doc_id}.json`
- `facts.db`

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Run

```bash
refinery "path/to/document.pdf"
```

or

```bash
python -m src.cli "path/to/document.pdf"
```

## Test

```bash
pytest
```

## Design notes

- Strategy A/B/C are implemented with confidence-gated escalation and budget guard.
- Provenance includes page + bbox + content hash.
- Query flow uses PageIndex navigation before semantic search.
- Fact extraction persists key-value signals to SQLite for structured queries.

## Current limitations

- Layout and vision extraction are implemented as extensible adapters/placeholders (no external VLM call by default).
- Semantic retrieval currently uses a local cosine-over-token baseline vector store.
- Section summaries are heuristic; swap in cheap LLM call if available.

## Next hardening steps

1. Integrate Docling or MinerU adapter for robust layout extraction.
2. Add OpenRouter-backed multimodal vision extractor with token accounting.
3. Add evaluation harness for retrieval precision and table extraction metrics.
4. Add Dockerfile and scripted demo workflow.
