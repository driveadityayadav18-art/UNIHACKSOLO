"""Evaluate a UniHack pipeline JSON result for demo metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from schema import EXPECTED_HEADERS, is_missing


def evaluate(result: Dict[str, Any]) -> Dict[str, Any]:
    records = result.get("records", [])
    traces = result.get("traceability", [])
    total_cells = max(1, len(records) * len(EXPECTED_HEADERS))
    populated = sum(1 for record in records for field in EXPECTED_HEADERS if not is_missing(record.get(field)))
    evidence_fields = sum(1 for trace in traces for meta in trace.get("fields", {}).values() if meta.get("status") in {"evidenced", "evidenced_enrichment"})
    source_urls = sum(1 for record in records for field in ["MFR URL", *[f"Ref URL {i}" for i in range(1, 6)]] if not is_missing(record.get(field)))
    hitl_fields = sum(len(trace.get("hitl_fields", [])) for trace in traces)
    missing_fields = sum(len(trace.get("missing_data", [])) for trace in traces)
    return {
        "records": len(records),
        "delivery_columns": len(EXPECTED_HEADERS),
        "schema_valid": bool(result.get("validation", {}).get("valid")),
        "populated_cell_coverage": round(populated / total_cells, 4),
        "evidence_backed_fields": evidence_fields,
        "source_urls": source_urls,
        "hitl_fields": hitl_fields,
        "missing_fields": missing_fields,
        "research_enabled": bool(result.get("research", {}).get("enabled")),
        "rows_researched": result.get("research", {}).get("rows_researched", 0),
        "research_evidence_items": result.get("research", {}).get("evidence_items", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Pipeline JSON output")
    args = parser.parse_args()
    result = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(json.dumps(evaluate(result), indent=2))


if __name__ == "__main__":
    main()
