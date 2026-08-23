"""FastAPI entrypoint for the UniHack product-intelligence pipeline."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from agents import Orchestrator, records_to_csv

app = FastAPI(
    title="UniHack 2026 Product Intelligence API",
    version="1.0.0",
    description="Ingests industrial product rows and returns expected-format records with traceability.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/process")
async def process(file: Optional[UploadFile] = File(default=None), text: Optional[str] = Form(default=None), use_llm: bool = Form(default=False), research_sources: bool = Form(default=False), research_limit: int = Form(default=5, ge=0, le=100)) -> JSONResponse:
    """Process a CSV, TXT, or PDF upload, or pasted text.

    The response contains `records` with exactly the 252 expected delivery keys
    and a separate `traceability` list so the CSV remains schema-compatible.
    """
    try:
        payload = await file.read() if file else None
        result = Orchestrator(use_llm=use_llm, research_sources=research_sources, research_limit=research_limit).process(payload=payload, filename=file.filename if file else "", text=text)
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


@app.post("/process/csv")
async def process_csv(file: UploadFile = File(...), use_llm: bool = Form(default=False), research_sources: bool = Form(default=False), research_limit: int = Form(default=5, ge=0, le=100)) -> PlainTextResponse:
    """Return only the exact expected-output CSV for downstream delivery."""
    try:
        payload = await file.read()
        result = Orchestrator(use_llm=use_llm, research_sources=research_sources, research_limit=research_limit).process(payload=payload, filename=file.filename or "upload.csv")
        return PlainTextResponse(
            records_to_csv(result["records"]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=unihack_product_output.csv"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/process/json")
async def process_json(file: Optional[UploadFile] = File(default=None), text: Optional[str] = Form(default=None), use_llm: bool = Form(default=False), research_sources: bool = Form(default=False), research_limit: int = Form(default=5, ge=0, le=100)) -> JSONResponse:
    """Alias that makes the JSON contract explicit for API clients."""
    return await process(file=file, text=text, use_llm=use_llm, research_sources=research_sources, research_limit=research_limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
