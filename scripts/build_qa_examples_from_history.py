from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ledger(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def _class_from_profile(profile: dict) -> str:
    origin = str(profile.get("origin_type", ""))
    layout = str(profile.get("layout_complexity", ""))
    domain = str(profile.get("domain_hint", ""))
    if origin == "scanned_image":
        return "B"
    if layout == "table_heavy":
        return "D"
    if domain == "financial":
        return "A"
    if domain == "technical":
        return "C"
    return "A"


def _class_from_document_name(name: str, fallback: str) -> str:
    low = (name or "").lower()
    if any(k in low for k in ["audit report", "assigned-regular-budget", "scanned"]):
        return "B"
    if any(k in low for k in ["tax_expenditure", "tax expenditure", "import tax"]):
        return "D"
    if any(k in low for k in ["annual report", "cbe", "financial report", "interim_trpw3"]):
        return "A"
    if any(k in low for k in ["trpw3_delivery", "technical", "performance_survey"]):
        return "C"
    return fallback


def _make_question(profile: dict, variant: int = 0) -> str:
    domain = str(profile.get("domain_hint", "general"))
    generic = [
        "What extraction strategy was selected for this document and why?",
        "What does the profile indicate about layout and extraction cost tier?",
        "What key section is present in this document according to extracted content?",
    ]
    if domain == "financial":
        q = [
            "What cost strategy is recommended for extracting this document?",
            "Is this document routed to fast text, layout, or vision and why?",
            "What financial extraction details are highlighted in the content?",
        ]
        return q[variant % len(q)]
    if domain == "technical":
        q = [
            "What is the document's extraction pipeline strategy and why?",
            "What does the extracted text say about the system architecture or pipeline?",
            "Which section in this technical document is most relevant to implementation?",
        ]
        return q[variant % len(q)]
    return generic[variant % len(generic)]


def _make_answer(profile: dict, ledger_row: dict | None, pageindex: dict | None) -> str:
    origin = profile.get("origin_type")
    layout = profile.get("layout_complexity")
    tier = profile.get("estimated_extraction_cost")
    strategy = (ledger_row or {}).get("strategy_used")
    trace = (ledger_row or {}).get("routing_trace") or []
    summary = ""
    if isinstance(pageindex, dict):
        children = pageindex.get("child_sections") or []
        if children and isinstance(children[0], dict):
            summary = str(children[0].get("summary", "")).strip()
    base = (
        f"The profile classifies the document as {origin} with {layout} layout "
        f"and estimated cost tier {tier}. "
        f"The executed strategy was {strategy} with routing trace {trace}."
    )
    if summary:
        return f"{base} Section summary: {summary}"
    return base


def _citations_from_chunks(chunks: list[dict], max_n: int = 3) -> list[dict]:
    out: list[dict] = []
    for ch in chunks:
        refs = ch.get("page_refs") or []
        if not refs:
            continue
        ref = refs[0]
        out.append(
            {
            "document_name": ref.get("document_name", ""),
            "page_number": int(ref.get("page_number", 0) or 0),
            "bbox": ref.get("bbox", {"x0": 0, "y0": 0, "x1": 0, "y1": 0}),
            "content_hash": ch.get("content_hash", ""),
            }
        )
        if len(out) >= max_n:
            break
    return out


def main() -> None:
    root = Path(".refinery")
    profiles_dir = root / "profiles"
    chunks_dir = root / "chunks"
    pageindex_dir = root / "pageindex"
    ledger_path = root / "extraction_ledger.jsonl"
    out_path = Path("artifacts/qa_examples.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not profiles_dir.exists() or not ledger_path.exists():
        raise SystemExit("Missing .refinery profiles/ledger; run pipeline first.")

    ledger = _load_ledger(ledger_path)
    by_doc_id: dict[str, dict] = {}
    for row in ledger:
        by_doc_id[str(row.get("doc_id"))] = row

    pools: dict[str, list[dict]] = {"A": [], "B": [], "C": [], "D": []}
    for prof_path in sorted(profiles_dir.glob("*.json")):
        doc_id = prof_path.stem
        chunk_path = chunks_dir / f"{doc_id}.json"
        if not chunk_path.exists():
            continue
        profile = _load_json(prof_path)
        chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
        if not chunks:
            continue
        ledger_row = by_doc_id.get(doc_id)
        pageindex_path = pageindex_dir / f"{doc_id}.json"
        pageindex = _load_json(pageindex_path) if pageindex_path.exists() else None

        citations = _citations_from_chunks(chunks, max_n=3)
        if not citations:
            continue

        cls = _class_from_document_name(profile.get("document_name", ""), _class_from_profile(profile))
        answer = _make_answer(profile, ledger_row, pageindex)
        pools.setdefault(cls, []).append(
            {
                "class": cls,
                "document_name": profile.get("document_name", ""),
                "doc_id": doc_id,
                "answer": answer,
                "citations": citations,
            }
        )

    # Emit exactly 3 per class when possible; allow repeated doc with different questions/citations.
    capped: list[dict] = []
    counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    for cls in ("A", "B", "C", "D"):
        candidates = pools.get(cls, [])
        if not candidates:
            continue
        i = 0
        while counts[cls] < 3 and i < 12:
            src = candidates[i % len(candidates)]
            cites = src["citations"]
            cit = cites[i % len(cites)]
            question = _make_question({"domain_hint": "financial" if cls in {"A", "D"} else "technical"}, variant=i)
            capped.append(
                {
                    "class": cls,
                    "document_name": src["document_name"],
                    "doc_id": src["doc_id"],
                    "question": question,
                    "answer": src["answer"],
                    "provenance_chain": {
                        "content_hash": cit["content_hash"],
                        "bbox": cit["bbox"],
                        "citations": [cit],
                    },
                    "verification_note": "Verify by opening the cited page and matching the extracted statement.",
                }
            )
            counts[cls] += 1
            i += 1

    payload = {
        "meta": {
            "description": "Q&A examples generated from .refinery history artifacts.",
            "counts_per_class": counts,
            "total_examples": len(capped),
            "source_artifacts": [
                ".refinery/extraction_ledger.jsonl",
                ".refinery/profiles/*.json",
                ".refinery/chunks/*.json",
                ".refinery/pageindex/*.json",
            ],
        },
        "examples": capped,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(capped)} history-derived examples to {out_path}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
