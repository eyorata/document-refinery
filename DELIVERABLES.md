# Final Submission Deliverables Checklist

## Repository

- [x] Stage 3 chunking agent: [src/agents/chunker.py](c:\Users\user\Documents\document-refinery\document-refinery\src\agents\chunker.py)
- [x] Stage 4 indexer agent: [src/agents/indexer.py](c:\Users\user\Documents\document-refinery\document-refinery\src\agents\indexer.py)
- [x] Stage 5 query agent: [src/agents/query_agent.py](c:\Users\user\Documents\document-refinery\document-refinery\src\agents\query_agent.py)
- [x] FactTable SQLite backend: [src/storage/fact_table.py](c:\Users\user\Documents\document-refinery\document-refinery\src\storage\fact_table.py)
- [x] Vector store ingestion (simple/FAISS): [src/storage/vector_store.py](c:\Users\user\Documents\document-refinery\document-refinery\src\storage\vector_store.py)
- [x] Audit mode (`verified | unverifiable`): [src/agents/query_agent.py](c:\Users\user\Documents\document-refinery\document-refinery\src\agents\query_agent.py)
- [x] Dockerfile: [Dockerfile](c:\Users\user\Documents\document-refinery\document-refinery\Dockerfile)

## Artifacts

- [x] PageIndex outputs: [.refinery/pageindex](c:\Users\user\Documents\document-refinery\document-refinery\.refinery\pageindex)
- [x] Profiles: [.refinery/profiles](c:\Users\user\Documents\document-refinery\document-refinery\.refinery\profiles)
- [x] Extraction ledger: [.refinery/extraction_ledger.jsonl](c:\Users\user\Documents\document-refinery\document-refinery\.refinery\extraction_ledger.jsonl)
- [x] Extracted docs JSON: [.refinery/extracted](c:\Users\user\Documents\document-refinery\document-refinery\.refinery\extracted)
- [x] Chunks JSON: [.refinery/chunks](c:\Users\user\Documents\document-refinery\document-refinery\.refinery\chunks)
- [x] PageIndex precision metrics: [.refinery/pageindex_metrics](c:\Users\user\Documents\document-refinery\document-refinery\.refinery\pageindex_metrics)
- [ ] 12 Q&A examples (3 per class) with full provenance:
  - Template: [artifacts/qa_examples.json](c:\Users\user\Documents\document-refinery\document-refinery\artifacts\qa_examples.json)
  - History auto-builder: [scripts/build_qa_examples_from_history.py](c:\Users\user\Documents\document-refinery\document-refinery\scripts\build_qa_examples_from_history.py)

## Evaluation Scripts

- [x] Table precision/recall evaluator:
  - [scripts/eval_table_metrics.py](c:\Users\user\Documents\document-refinery\document-refinery\scripts\eval_table_metrics.py)

Run:

```bash
python scripts/eval_table_metrics.py
```

Build Q&A examples from existing `.refinery` history:

```bash
python scripts/build_qa_examples_from_history.py
```

Optionally pass a gold file:

```bash
python scripts/eval_table_metrics.py --gold artifacts/table_gold_labels.json
```

## Final Report

- [x] Draft report: [FINAL_REPORT_DRAFT.md](c:\Users\user\Documents\document-refinery\document-refinery\FINAL_REPORT_DRAFT.md)
- [ ] Export as a single PDF for submission
