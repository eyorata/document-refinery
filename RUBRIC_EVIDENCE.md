# Rubric Evidence Map

## 1) Extraction Fidelity

- Multi-strategy extraction + escalation:
  - [src/agents/extractor.py](c:\Users\user\Documents\document-refinery\document-refinery\src\agents\extractor.py)
  - [src/agents/orchestrator.py](c:\Users\user\Documents\document-refinery\document-refinery\src\agents\orchestrator.py)
- Strategy A/B/C implementations:
  - [src/strategies/fast_text.py](c:\Users\user\Documents\document-refinery\document-refinery\src\strategies\fast_text.py)
  - [src/strategies/layout_aware.py](c:\Users\user\Documents\document-refinery\document-refinery\src\strategies\layout_aware.py)
  - [src/strategies/vision.py](c:\Users\user\Documents\document-refinery\document-refinery\src\strategies\vision.py)
- Extraction logs (confidence, strategy, provider, token usage):
  - [.refinery/extraction_ledger.jsonl](c:\Users\user\Documents\document-refinery\document-refinery\.refinery\extraction_ledger.jsonl)
- Extracted structured tables:
  - [.refinery/extracted](c:\Users\user\Documents\document-refinery\document-refinery\.refinery\extracted)

## 2) Architecture Quality

- Typed pipeline + schemas:
  - [src/pipeline.py](c:\Users\user\Documents\document-refinery\document-refinery\src\pipeline.py)
  - [src/models/schemas.py](c:\Users\user\Documents\document-refinery\document-refinery\src\models\schemas.py)
- Config-driven routing/budgets:
  - [rubric/extraction_rules.yaml](c:\Users\user\Documents\document-refinery\document-refinery\rubric\extraction_rules.yaml)
  - [src/config.py](c:\Users\user\Documents\document-refinery\document-refinery\src\config.py)
- Tests:
  - `pytest -q`

## 3) Provenance & Indexing

- Chunking + validator + content hashes:
  - [src/agents/chunker.py](c:\Users\user\Documents\document-refinery\document-refinery\src\agents\chunker.py)
- PageIndex builder + precision evaluation:
  - [src/agents/indexer.py](c:\Users\user\Documents\document-refinery\document-refinery\src\agents\indexer.py)
  - [.refinery/pageindex](c:\Users\user\Documents\document-refinery\document-refinery\.refinery\pageindex)
  - [.refinery/pageindex_metrics](c:\Users\user\Documents\document-refinery\document-refinery\.refinery\pageindex_metrics)
- Query agent with provenance chain:
  - [src/agents/query_agent.py](c:\Users\user\Documents\document-refinery\document-refinery\src\agents\query_agent.py)

## 4) Data Layer

- FactTable SQLite:
  - [src/storage/fact_table.py](c:\Users\user\Documents\document-refinery\document-refinery\src\storage\fact_table.py)
  - [.refinery/facts.db](c:\Users\user\Documents\document-refinery\document-refinery\.refinery\facts.db)
- Vector store ingestion (`simple`/`faiss`):
  - [src/storage/vector_store.py](c:\Users\user\Documents\document-refinery\document-refinery\src\storage\vector_store.py)

## 5) Domain Onboarding

- Domain analysis + decision tree + failure modes:
  - [DOMAIN_NOTES.md](c:\Users\user\Documents\document-refinery\document-refinery\DOMAIN_NOTES.md)

## 6) Final Submission Artifacts

- Report draft:
  - [FINAL_REPORT_DRAFT.md](c:\Users\user\Documents\document-refinery\document-refinery\FINAL_REPORT_DRAFT.md)
- Q&A examples with provenance:
  - [artifacts/qa_examples.json](c:\Users\user\Documents\document-refinery\document-refinery\artifacts\qa_examples.json)
- Table metrics tooling:
  - [scripts/eval_table_metrics.py](c:\Users\user\Documents\document-refinery\document-refinery\scripts\eval_table_metrics.py)
  - [.refinery/table_metrics.json](c:\Users\user\Documents\document-refinery\document-refinery\.refinery\table_metrics.json)
- Docker:
  - [Dockerfile](c:\Users\user\Documents\document-refinery\document-refinery\Dockerfile)

