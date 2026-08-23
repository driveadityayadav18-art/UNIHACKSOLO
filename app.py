"""Streamlit HITL review application for the UniHack pipeline."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from agents import Orchestrator, records_to_csv
from evaluate import evaluate
from schema import EXPECTED_HEADERS

st.set_page_config(page_title="UniHack Product Intelligence", page_icon="", layout="wide")
st.title("UniHack 2026 — Product Intelligence Review")
st.caption("Upload a catalog or paste product text. Generated records preserve the exact 252-column delivery format; low-confidence fields are highlighted for review.")

with st.sidebar:
    st.header("Pipeline controls")
    use_llm = st.checkbox("Enable optional Groq AI enrichment", value=False, help="Uses GROQ_API_KEY and GROQ_MODEL when enabled. Keep off for instant deterministic local processing.")
    llm_limit = st.number_input("Rows for AI enrichment", min_value=1, max_value=100, value=5, step=1, disabled=not use_llm, help="Enriches the first N rows with Groq AI so batch uploads complete quickly.")
    research_sources = st.checkbox("Research public sources", value=False, help="Searches public web results and retrieves readable HTML/PDF evidence. Start with a small limit.")
    research_limit = st.number_input("Rows to research", min_value=0, max_value=100, value=5, step=1, disabled=not research_sources)
    st.divider()
    st.info("💡 **Performance Note:** Offline deterministic mode processes all 1,000 rows in ~1 second. AI Enrichment & Public Research are applied to the first N rows selected above.")

uploaded = st.file_uploader("Upload CSV, TXT, or PDF", type=["csv", "txt", "pdf"])
text_input = st.text_area("Or paste raw product text", height=150, placeholder="Example: 3M 775L Stikit Film P150 - Cubitron II 50 Disc/Box")

if st.button("Process products", type="primary"):
    if not uploaded and not text_input.strip():
        st.error("Upload a file or paste product text first.")
    else:
        try:
            progress_holder = st.empty()
            progress_bar = progress_holder.progress(0, text="Starting ingestion and extraction...")
            
            def on_progress(current: int, total: int):
                pct = min(1.0, current / max(1, total))
                progress_bar.progress(pct, text=f"Processing row {current:,} of {total:,}...")

            payload = uploaded.getvalue() if uploaded else None
            filename = uploaded.name if uploaded else "text-input.txt"
            orchestrator = Orchestrator(
                use_llm=use_llm,
                llm_limit=int(llm_limit),
                research_sources=research_sources,
                research_limit=int(research_limit)
            )
            result = orchestrator.process(
                payload=payload,
                filename=filename,
                text=text_input or None,
                progress_callback=on_progress
            )
            progress_holder.empty()
            st.session_state["result"] = result
            st.success(f"Successfully processed {len(result['records']):,} product row(s) into 252 delivery columns.")
        except Exception as exc:
            st.exception(exc)

result: Dict[str, Any] = st.session_state.get("result", {})
if result:
    validation = result.get("validation", {})
    records: List[Dict[str, Any]] = result.get("records", [])
    traces: List[Dict[str, Any]] = result.get("traceability", [])
    metrics = evaluate(result)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Records", f"{len(records):,}")
    metric_cols[1].metric("Delivery columns", len(EXPECTED_HEADERS))
    metric_cols[2].metric("Rows needing review", sum(bool(item.get("hitl_fields")) for item in traces))
    metric_cols[3].metric("Evidence items", metrics["research_evidence_items"])
    metric_cols[4].metric("Schema valid", "Yes" if validation.get("valid") else "No")

    if validation.get("errors"):
        st.error("\n".join(validation["errors"][:20]))
    if validation.get("warnings"):
        st.warning("\n".join(validation["warnings"][:10]))

    st.subheader("Review records")
    if records:
        record_index = st.number_input("Record number", min_value=1, max_value=len(records), value=1, step=1) - 1
        selected_record = records[record_index]
        selected_trace = traces[record_index] if record_index < len(traces) else {"fields": {}}
        trace_fields = selected_trace.get("fields", {})
        rows = []
        for field in EXPECTED_HEADERS:
            meta = trace_fields.get(field, {})
            rows.append({
                "Field": field,
                "Value": selected_record.get(field) or "",
                "Confidence": meta.get("confidence_score", 0.0),
                "Status": meta.get("status", "missing"),
                "Source": meta.get("source", "not available"),
                "Reasoning": meta.get("reasoning", ""),
            })
        review_df = pd.DataFrame(rows)

        def highlight_low_confidence(row: pd.Series) -> List[str]:
            color = "background-color: #fff3cd" if float(row["Confidence"]) < 0.8 else ""
            return [color] * len(row)

        st.dataframe(review_df.style.apply(highlight_low_confidence, axis=1).format({"Confidence": "{:.2f}"}), use_container_width=True, height=650)
        low_fields = selected_trace.get("hitl_fields", [])
        if low_fields:
            st.caption("Yellow rows have confidence below 0.80 and should be reviewed before publication.")
            with st.expander(f"HITL queue ({len(low_fields)} fields)"):
                st.write(", ".join(low_fields))

    st.subheader("Download")
    col_json, col_csv = st.columns(2)
    with col_json:
        st.download_button("Download JSON with traceability", data=json.dumps(result, indent=2, ensure_ascii=False), file_name="unihack_product_output.json", mime="application/json")
    with col_csv:
        st.download_button("Download exact delivery CSV", data=records_to_csv(records), file_name="unihack_product_output.csv", mime="text/csv")

    research = result.get("research", {})
    if research.get("enabled"):
        with st.expander(f"Source research details ({research.get('rows_researched', 0)} row(s) researched)"):
            st.json(research)

    with st.expander("Evaluation metrics"):
        st.json(metrics)

    with st.expander("Delivery schema"):
        st.write(EXPECTED_HEADERS)
