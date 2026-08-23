"""Generate and validate sample output artifacts for the supplied dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents import Orchestrator, records_to_csv
from schema import EXPECTED_HEADERS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to sample input CSV")
    parser.add_argument("--json-output", default="sample_output.json")
    parser.add_argument("--csv-output", default="sample_output.csv")
    parser.add_argument("--research", action="store_true", help="Research public sources for the first N rows")
    parser.add_argument("--research-limit", type=int, default=5)
    args = parser.parse_args()

    source = Path(args.input)
    result = Orchestrator(use_llm=False, research_sources=args.research, research_limit=args.research_limit).process(payload=source.read_bytes(), filename=source.name)
    csv_text = records_to_csv(result["records"])
    header = csv_text.splitlines()[0].split(",")
    assert header == EXPECTED_HEADERS, f"Header mismatch: {len(header)} columns returned"
    assert len(result["records"]) == result["ingestion"]["row_count"]
    assert all(len(item["fields"]) + len(item["missing_data"]) == len(EXPECTED_HEADERS) for item in result["traceability"])

    Path(args.json_output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.csv_output).write_text(csv_text, encoding="utf-8", newline="")
    print(json.dumps({
        "rows": len(result["records"]),
        "columns": len(header),
        "schema_valid": result["validation"]["valid"],
        "hitl_records": sum(bool(item["hitl_fields"]) for item in result["traceability"]),
        "research_enabled": result["research"]["enabled"],
        "rows_researched": result["research"]["rows_researched"],
        "evidence_items": result["research"]["evidence_items"],
        "json_output": str(Path(args.json_output).resolve()),
        "csv_output": str(Path(args.csv_output).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
