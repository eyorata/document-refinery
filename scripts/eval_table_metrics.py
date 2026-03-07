from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_with_gold(extracted_dir: Path, gold_path: Path) -> tuple[Counts, list[dict]]:
    """
    Gold schema:
    {
      "items": [
        {"doc_id":"...", "has_table": true},
        {"document_name":"file.pdf", "has_table": false}
      ]
    }
    """
    gold = load_json(gold_path)
    items = gold.get("items", [])
    by_doc_id: dict[str, bool] = {}
    by_name: dict[str, bool] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        has_table = bool(it.get("has_table", False))
        if it.get("doc_id"):
            by_doc_id[str(it["doc_id"])] = has_table
        if it.get("document_name"):
            by_name[str(it["document_name"])] = has_table

    counts = Counts()
    details: list[dict] = []
    for f in sorted(extracted_dir.glob("*.json")):
        d = load_json(f)
        doc_id = f.stem
        name = str(d.get("document_name", ""))
        pred = bool(d.get("tables"))
        if doc_id in by_doc_id:
            gt = by_doc_id[doc_id]
        elif name in by_name:
            gt = by_name[name]
        else:
            continue
        if gt and pred:
            counts.tp += 1
        elif (not gt) and pred:
            counts.fp += 1
        elif gt and (not pred):
            counts.fn += 1
        else:
            counts.tn += 1
        details.append(
            {
                "doc_id": doc_id,
                "document_name": name,
                "ground_truth_has_table": gt,
                "pred_has_table": pred,
                "pred_table_count": len(d.get("tables", [])),
            }
        )
    return counts, details


def evaluate_proxy(extracted_dir: Path, profiles_dir: Path) -> tuple[Counts, list[dict]]:
    """
    Proxy evaluation when gold labels are absent:
    - Ground truth proxy = profile.layout_complexity == table_heavy
    - Prediction = extracted.tables not empty
    """
    counts = Counts()
    details: list[dict] = []
    for f in sorted(extracted_dir.glob("*.json")):
        d = load_json(f)
        doc_id = f.stem
        p = profiles_dir / f"{doc_id}.json"
        if not p.exists():
            continue
        profile = load_json(p)
        gt = str(profile.get("layout_complexity", "")) == "table_heavy"
        pred = bool(d.get("tables"))
        if gt and pred:
            counts.tp += 1
        elif (not gt) and pred:
            counts.fp += 1
        elif gt and (not pred):
            counts.fn += 1
        else:
            counts.tn += 1
        details.append(
            {
                "doc_id": doc_id,
                "document_name": d.get("document_name"),
                "ground_truth_proxy_table_heavy": gt,
                "pred_has_table": pred,
                "pred_table_count": len(d.get("tables", [])),
                "layout_complexity": profile.get("layout_complexity"),
            }
        )
    return counts, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate table extraction precision/recall from refinery artifacts.")
    parser.add_argument("--refinery-dir", default=".refinery", help="Artifact root directory")
    parser.add_argument("--gold", default="", help="Optional gold labels JSON path")
    parser.add_argument(
        "--output",
        default=".refinery/table_metrics.json",
        help="Output JSON file for aggregate metrics and per-doc details",
    )
    args = parser.parse_args()

    root = Path(args.refinery_dir)
    extracted_dir = root / "extracted"
    profiles_dir = root / "profiles"
    if not extracted_dir.exists():
        raise SystemExit(f"Missing extracted directory: {extracted_dir}")

    if args.gold:
        counts, details = evaluate_with_gold(extracted_dir, Path(args.gold))
        mode = "gold"
    else:
        counts, details = evaluate_proxy(extracted_dir, profiles_dir)
        mode = "proxy"

    out = {
        "mode": mode,
        "counts": {
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
            "tn": counts.tn,
            "n": len(details),
        },
        "metrics": {
            "precision": counts.precision(),
            "recall": counts.recall(),
            "f1": counts.f1(),
        },
        "details": details,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out["counts"], indent=2))
    print(json.dumps(out["metrics"], indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

