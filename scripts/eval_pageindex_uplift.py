from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.indexer import PageIndexBuilder
from src.models import LDU, PageIndexNode
from src.storage.vector_store import SimpleVectorStore


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ldus(path: Path) -> list[LDU]:
    data = _load_json(path)
    return [LDU.model_validate(item) for item in data]


def _load_pageindex(path: Path) -> PageIndexNode:
    return PageIndexNode.model_validate(_load_json(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PageIndex retrieval uplift from saved artifacts.")
    parser.add_argument("--refinery-dir", default=".refinery")
    parser.add_argument("--queries", default="artifacts/pageindex_eval_queries.json")
    parser.add_argument("--output", default=".refinery/pageindex_uplift_report.json")
    args = parser.parse_args()

    root = Path(args.refinery_dir)
    chunks_dir = root / "chunks"
    pageindex_dir = root / "pageindex"
    query_cfg = Path(args.queries)
    if not chunks_dir.exists() or not pageindex_dir.exists():
        raise SystemExit("Missing chunks/pageindex artifacts.")
    if not query_cfg.exists():
        raise SystemExit(f"Missing query config file: {query_cfg}")

    cfg = _load_json(query_cfg)
    items = cfg.get("items", [])
    builder = PageIndexBuilder(pageindex_cfg={"llm_summaries_enabled": False, "llm": {"enabled": False}})

    rows: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        doc_id = str(it.get("doc_id", "")).strip()
        query = str(it.get("query", "")).strip()
        if not doc_id or not query:
            continue
        chunks_path = chunks_dir / f"{doc_id}.json"
        index_path = pageindex_dir / f"{doc_id}.json"
        if not chunks_path.exists() or not index_path.exists():
            continue
        chunks = _load_ldus(chunks_path)
        index = _load_pageindex(index_path)
        vs = SimpleVectorStore()
        vs.ingest(chunks)
        m = builder.evaluate_retrieval_precision(
            topic=query,
            chunks=chunks,
            search_fn=vs.search,
            page_index=index,
            top_k=int(it.get("top_k", 3) or 3),
        )
        rows.append(
            {
                "doc_id": doc_id,
                "query": query,
                "with_pageindex": m.get("with_pageindex", 0.0),
                "without_pageindex": m.get("without_pageindex", 0.0),
                "delta": m.get("with_pageindex", 0.0) - m.get("without_pageindex", 0.0),
            }
        )

    avg_with = sum(r["with_pageindex"] for r in rows) / len(rows) if rows else 0.0
    avg_without = sum(r["without_pageindex"] for r in rows) / len(rows) if rows else 0.0
    out = {
        "n": len(rows),
        "avg_with_pageindex": avg_with,
        "avg_without_pageindex": avg_without,
        "avg_delta": avg_with - avg_without,
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
