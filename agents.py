"""UniHack multi-agent product intelligence pipeline.

The implementation is deliberately resilient: it runs without an API key using
transparent deterministic extraction and a local similarity knowledge base,
and can optionally call OpenAI through LangChain for enrichment.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

from schema import (
    AIEnrichmentResponse,
    EXPECTED_HEADERS,
    INPUT_HEADERS,
    FieldTrace,
    ProductRecord,
    ProductTraceability,
    ValidationReport,
    is_missing,
    model_to_delivery_dict,
)
from source_research import (
    DocumentIngestionAgent,
    LocalDocumentRAG,
    SourceDiscoveryAgent,
    find_evidence,
)

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional at runtime
    PdfReader = None


def build_structured_chat_model():
    """Create a configured Groq or OpenAI model using environment variables only."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class IngestionResult:
    rows: List[Dict[str, str]]
    raw_text: str
    filename: str = ""
    warnings: List[str] = field(default_factory=list)


class IngestionAgent:
    """Normalizes CSV, text, and PDF inputs into source rows."""

    def ingest(self, payload: Optional[bytes] = None, filename: str = "", text: Optional[str] = None) -> IngestionResult:
        if text and text.strip():
            return self._from_text(text, filename or "text-input.txt")
        if not payload:
            raise ValueError("Provide either text or a non-empty uploaded file")

        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            if PdfReader is None:
                raise ValueError("PDF support requires pypdf; install requirements.txt")
            reader = PdfReader(io.BytesIO(payload))
            raw = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            return self._from_text(raw, filename)
        decoded = payload.decode("utf-8-sig", errors="replace")
        if suffix in {".csv", ".tsv"} or "," in decoded.splitlines()[0] if decoded.splitlines() else False:
            delimiter = "\t" if suffix == ".tsv" else ","
            reader = csv.DictReader(io.StringIO(decoded), delimiter=delimiter)
            rows = [{str(k): (v or "").strip() for k, v in row.items() if k is not None} for row in reader]
            if rows:
                return IngestionResult(rows=rows, raw_text=decoded, filename=filename)
        return self._from_text(decoded, filename or "uploaded.txt")

    def _from_text(self, text: str, filename: str) -> IngestionResult:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("The supplied text/file did not contain readable text")
        rows: List[Dict[str, str]] = []
        for line in lines:
            rows.append({"Part_Desc": line, "Mfg_Part_Num": self._guess_part_number(line)})
        return IngestionResult(rows=rows, raw_text=text, filename=filename)

    @staticmethod
    def _guess_part_number(text: str) -> str:
        match = re.search(r"\b[A-Z0-9][A-Z0-9._/-]{3,}\b", text, flags=re.I)
        return match.group(0) if match else ""


class LocalKnowledgeBase:
    """Small local RAG index over the rest of the supplied catalog.

    The default implementation is dependency-free and uses token overlap. If
    FAISS is installed, the class can be replaced by a FAISS-backed adapter
    without changing the agents' interface.
    """

    def __init__(self, rows: Optional[Iterable[Dict[str, str]]] = None):
        self.documents: List[Tuple[str, Dict[str, str]]] = []
        for row in rows or []:
            text = " ".join(str(row.get(key, "")) for key in ("Mfg_Part_Num", "Part_Desc", "Part_Manuf"))
            if text.strip():
                self.documents.append((text, row))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def search(self, query: str, k: int = 3) -> List[Tuple[float, Dict[str, str]]]:
        query_tokens = self._tokens(query)
        scored: List[Tuple[float, Dict[str, str]]] = []
        for text, row in self.documents:
            tokens = self._tokens(text)
            score = len(query_tokens & tokens) / max(1, len(query_tokens | tokens))
            scored.append((score, row))
        return sorted(scored, key=lambda item: item[0], reverse=True)[:k]


class ExtractionAgent:
    """Maps source fields and product-description signals to the flat schema."""

    KNOWN_BRANDS = {
        "3m": "3M", "diablo": "Diablo", "mirka": "Mirka", "milw": "Milwaukee",
        "milwaukee": "Milwaukee", "makita": "Makita", "dewalt": "DEWALT", "black": "BLACK+DECKER",
        "kreg": "Kreg", "festool": "Festool", "satco": "Satco", "kichler": "Kichler",
        "leviton": "Leviton", "southwire": "Southwire", "phillips": "Philips",
    }

    def extract(self, source: Dict[str, str]) -> Tuple[Dict[str, Optional[str]], Dict[str, FieldTrace]]:
        desc = self._clean(source.get("Part_Desc", ""))
        part = self._clean(source.get("Mfg_Part_Num", ""))
        manufacturer_raw = self._clean(source.get("Part_Manuf", ""))
        manufacturer = re.sub(r"\s*\([^)]*\)\s*$", "", manufacturer_raw).strip(" -")
        brand = self._brand(source, desc, manufacturer)
        record: Dict[str, Optional[str]] = {header: None for header in EXPECTED_HEADERS}
        traces: Dict[str, FieldTrace] = {}

        direct = {key: self._clean(source.get(key, "")) for key in INPUT_HEADERS}
        for key, value in direct.items():
            if value:
                record[key] = value
                traces[key] = self._trace(value, "input", value, "Copied directly from the uploaded source row.", 1.0, "observed")

        inferred: Dict[str, Tuple[Optional[str], str, float, str]] = {
            "MANUFACTURER_NAME": (manufacturer or None, "manufacturer parsed from Part_Manuf", 0.94, "Removed the optional parenthesized supplier code from Part_Manuf."),
            "BRAND_NAME": (brand, "brand inferred from description/manufacturer", 0.84 if brand else 0.2, "Matched a known industrial brand token in the description or manufacturer."),
            "MANUFACTURER_PART_NUMBER": (part or None, "Mfg_Part_Num", 0.99, "The manufacturer part number is the source Mfg_Part_Num."),
            "MOBILE_DESC": (desc or None, "Part_Desc", 0.96, "Reused the supplied description because no external product copy was provided."),
            "SHORT_DESC": (desc or None, "Part_Desc", 0.96, "Used the source description as the conservative short description."),
            "RETAIL_DESC": (desc or None, "Part_Desc", 0.86, "Used the source description without adding unsupported claims."),
            "Product Name": (self._product_name(desc) or None, "Part_Desc", 0.84, "Normalized the description into a concise product name."),
            "Classpath": (self._classpath(desc), "description taxonomy heuristic", 0.72, "Assigned a broad industrial taxonomy from product terms; review if taxonomy precision is critical."),
        }
        for key, (value, source_name, score, reasoning) in inferred.items():
            if value:
                record[key] = value
                traces[key] = self._trace(value, "inference", source_name, reasoning, score, "inferred")

        for idx, feature in enumerate(self._features(desc), start=1):
            key = f"ITEM_FEATURES_{idx}"
            record[key] = feature
            traces[key] = self._trace(feature, "Part_Desc", desc, "Extracted a compact feature phrase from the supplied description.", 0.78, "inferred")

        for idx, (label, value, uom) in enumerate(self._attributes(desc), start=1):
            for key, val in ((f"ATTRIBUTE_LABEL {idx}", label), (f"ATTRIBUTE_VALUE {idx}", value), (f"ATTRIBUTE_UOM {idx}", uom)):
                record[key] = val or None
                traces[key] = self._trace(val, "Part_Desc", desc, "Parsed a structured attribute from a dimension, grit, pack, or thread pattern.", 0.82, "inferred")

        return record, traces

    def _brand(self, source: Dict[str, str], desc: str, manufacturer: str) -> Optional[str]:
        for value in (source.get("E1_Brand", ""), source.get("Unilog_Brand", ""), source.get("DIB_Brand", "")):
            if not is_missing(value):
                return value.strip()
        haystack = f"{desc} {manufacturer}".lower()
        for token, name in self.KNOWN_BRANDS.items():
            if re.search(rf"\b{re.escape(token)}\b", haystack):
                return name
        return None

    @staticmethod
    def _clean(value: Any) -> str:
        value = "" if value is None else str(value)
        return " ".join(value.replace("\u0000", "").split()).strip()

    @staticmethod
    def _product_name(desc: str) -> str:
        return re.sub(r"^([A-Z0-9][A-Z0-9._/-]{3,})\s+", "", desc, flags=re.I).strip(" -")[:180]

    @staticmethod
    def _classpath(desc: str) -> Optional[str]:
        lower = desc.lower()
        if any(term in lower for term in ("dishwasher", "refrigerator", "range", "oven")):
            return "Appliances & Consumer Electronics>Kitchen Appliances"
        if any(term in lower for term in ("cut off disc", "grinding", "sanding", "abrasive", "sanding belt")):
            return "Tools & Equipment>Power Tool Accessories>Abrasives"
        if "light" in lower or "lamp" in lower:
            return "Electrical>Lighting"
        if "wire" in lower or "cable" in lower:
            return "Electrical>Wire & Cable"
        return "Industrial Supplies>General"

    @staticmethod
    def _features(desc: str) -> List[str]:
        candidates: List[str] = []
        patterns = [
            r"\bP\d+\b", r"\b\d+(?:\.\d+)?\s*(?:pc|pcs|piece|pieces|disc|discs|box)\b",
            r"\b\d+(?:-\d+)?(?:\.\d+)?\s*(?:in|mm|cm)\b(?:\s*x\s*[^,;]+)?",
            r"\b(?:steel|metal|masonry|ceramic|film|abrasive|stainless steel|aluminum|aluminium)\b",
            r"\b(?:cut[- ]off|cut and grind|sanding belt|disc|wheel|abrasive)\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, desc, flags=re.I):
                phrase = match.group(0).strip()
                if phrase.lower() not in {item.lower() for item in candidates}:
                    candidates.append(phrase)
        return candidates[:20]

    @staticmethod
    def _attributes(desc: str) -> List[Tuple[str, str, str]]:
        result: List[Tuple[str, str, str]] = []
        for match in re.finditer(r"\b(P\d+)\b", desc, flags=re.I):
            result.append(("Grit", match.group(1).upper(), ""))
        for match in re.finditer(r"(?<!\w)(\d+(?:\.\d+)?(?:-\d+)?(?:/\d+)?)(?:\s*)(in|mm|cm)(?!\w)", desc, flags=re.I):
            result.append(("Size", match.group(1), match.group(2).lower()))
        pack = re.search(r"\b(\d+)\s*(?:pc|pcs|piece|pieces)\b", desc, flags=re.I)
        if pack:
            result.append(("Pieces per Package", pack.group(1), "EA"))
        thread = re.search(r"\b(\d+(?:/\d+)?-\d+)\b", desc)
        if thread:
            result.append(("Thread Size", thread.group(1), ""))
        return result[:50]

    @staticmethod
    def _trace(value: Optional[str], source: str, excerpt: str, reasoning: str, score: float, status: str) -> FieldTrace:
        return FieldTrace(value=value, confidence_score=max(0.0, min(1.0, score)), source=source, source_excerpt=excerpt[:240], reasoning=reasoning, status=status)


class EnrichmentAgent:
    """Adds conservative local-KB or optional LLM enrichments for missing fields."""

    def __init__(self, knowledge_base: Optional[LocalKnowledgeBase] = None, use_llm: Optional[bool] = None):
        self.knowledge_base = knowledge_base or LocalKnowledgeBase()
        self.use_llm = bool(use_llm) if use_llm is not None else os.getenv("USE_LLM_ENRICHMENT", "false").lower() == "true"

    def enrich(self, record: Dict[str, Optional[str]], traces: Dict[str, FieldTrace], source: Dict[str, str]) -> Tuple[Dict[str, Optional[str]], Dict[str, FieldTrace]]:
        query = " ".join(str(source.get(key, "")) for key in INPUT_HEADERS)
        neighbors = self.knowledge_base.search(query, k=3)
        nearest = neighbors[0][1] if neighbors and neighbors[0][0] >= 0.15 else None

        # Reuse only catalog facts that are safe and directly applicable.
        if is_missing(record.get("BRAND_NAME")) and nearest and not is_missing(nearest.get("E1_Brand")):
            value = nearest.get("E1_Brand")
            record["BRAND_NAME"] = value
            traces["BRAND_NAME"] = FieldTrace(value=value, confidence_score=0.66, source="local knowledge base", source_excerpt=str(nearest)[:240], reasoning="Borrowed a brand only from the nearest catalog row; flagged for human review because similarity is indirect.", status="enriched")

        if self.use_llm:
            self._optional_llm_enrich(record, traces, source)
        return record, traces

    def _optional_llm_enrich(self, record: Dict[str, Optional[str]], traces: Dict[str, FieldTrace], source: Dict[str, str]) -> None:
        """Call the selected LangChain provider only when explicitly enabled."""
        try:
            from langchain_core.messages import HumanMessage
            chat_model = build_structured_chat_model()
        except ImportError:
            return
        missing = [key for key in ("MARKETING_DESCRIPTION", "LONG_DESC1", "Application", "Includes") if is_missing(record.get(key))]
        if not missing:
            return
        prompt = {
            "task": "Generate conservative industrial product metadata from the supplied row. Never invent numeric specifications or unsupported claims.",
            "fields": missing,
            "source": {key: source.get(key, "") for key in INPUT_HEADERS},
            "output_contract": "Return an object with a fields map. Each requested field must contain value, confidence_score, reasoning, and optional source_url.",
        }
        try:
            structured = chat_model.with_structured_output(AIEnrichmentResponse)
            response = structured.invoke([HumanMessage(content=json.dumps(prompt))])
            parsed = response if isinstance(response, AIEnrichmentResponse) else AIEnrichmentResponse.model_validate(response)
        except Exception:
            return
        for key, suggestion in parsed.fields.items():
            if key not in missing or is_missing(suggestion.value):
                continue
            value = str(suggestion.value).strip()
            source_name = suggestion.source_url or "OpenAI via LangChain"
            record[key] = value
            traces[key] = FieldTrace(value=value, confidence_score=min(suggestion.confidence_score, 0.79), source=source_name, source_excerpt=str(source)[:240], reasoning=suggestion.reasoning, status="enriched")


class SourceResearchAgent:
    """Discovers public sources, fetches readable documents, and attaches evidence."""

    def __init__(self, max_results: int = 3, use_llm: bool = False):
        self.discovery = SourceDiscoveryAgent()
        self.ingestion = DocumentIngestionAgent()
        self.max_results = max_results
        self.use_llm = use_llm

    def enrich_from_sources(self, record: Dict[str, Optional[str]], traces: Dict[str, FieldTrace], source: Dict[str, str]) -> Tuple[Dict[str, Optional[str]], Dict[str, FieldTrace], Dict[str, Any]]:
        references, warnings = self.discovery.discover(source, max_results=self.max_results)
        manufacturer_tokens = [token for token in re.findall(r"[a-z0-9]+", str(source.get("Part_Manuf", "")).lower()) if token not in {"inc", "llc", "corp", "corporation", "co", "company"} and len(token) > 2]
        manufacturer_url_used = False
        ref_index = 1
        for reference in references:
            domain = urlparse(reference.url).netloc.lower()
            is_manufacturer_url = bool(manufacturer_tokens and any(token in domain for token in manufacturer_tokens))
            if is_manufacturer_url and not manufacturer_url_used:
                key = "MFR URL"
                manufacturer_url_used = True
            else:
                key = f"Ref URL {ref_index}"
                ref_index += 1
            record[key] = reference.url
            traces[key] = FieldTrace(value=reference.url, confidence_score=0.78 if is_manufacturer_url else 0.68, source="public source discovery", source_excerpt=reference.title[:240], reasoning="Discovered as a candidate public source; manufacturer URL is assigned only when the domain matches a manufacturer token, otherwise it remains a reference URL.", status="source_candidate")
        documents = [self.ingestion.fetch(reference) for reference in references]
        rag = LocalDocumentRAG(documents)
        evidence_count = 0
        query = " ".join(str(source.get(key, "")) for key in INPUT_HEADERS)
        for field in ("MANUFACTURER_NAME", "BRAND_NAME", "MANUFACTURER_PART_NUMBER", "Product Name", "MOBILE_DESC", "SHORT_DESC"):
            value = record.get(field)
            if is_missing(value):
                continue
            evidence = find_evidence(rag, f"{query} {field} {value}", field, str(value))
            if evidence is None:
                continue
            evidence_count += 1
            traces[field] = FieldTrace(value=str(value), confidence_score=evidence.confidence_score, source=evidence.source_url, source_excerpt=evidence.excerpt, reasoning=evidence.reasoning, status="evidenced")
        if self.use_llm and rag.chunks:
            self._llm_evidence_enrich(record, traces, source, rag)
        return record, traces, {
            "references": [reference.__dict__ for reference in references],
            "documents": [{"url": document.url, "title": document.title, "content_type": document.content_type, "chars": len(document.text), "error": document.error} for document in documents],
            "evidence_count": evidence_count,
            "warnings": warnings + [document.error for document in documents if document.error],
        }

    def _llm_evidence_enrich(self, record: Dict[str, Optional[str]], traces: Dict[str, FieldTrace], source: Dict[str, str], rag: LocalDocumentRAG) -> None:
        """Use retrieved passages as grounding context for structured enrichment."""
        try:
            from langchain_core.messages import HumanMessage
            chat_model = build_structured_chat_model()
        except ImportError:
            return
        target_fields = [key for key in ("LONG_DESC1", "MARKETING_DESCRIPTION", "Application", "Includes", "Standard/Approvals", "Warranty") if is_missing(record.get(key))]
        if not target_fields:
            return
        query = " ".join(str(source.get(key, "")) for key in INPUT_HEADERS)
        passages = rag.search(query, k=5)
        evidence_context = "\n\n".join(f"SOURCE: {document.url}\nEXCERPT: {excerpt}" for _, document, excerpt in passages)
        prompt = {
            "task": "Fill only requested commerce fields from the supplied source excerpts. Do not infer unsupported values; return null when evidence is absent.",
            "requested_fields": target_fields,
            "source_row": {key: source.get(key, "") for key in INPUT_HEADERS},
            "evidence": evidence_context,
        }
        try:
            structured = chat_model.with_structured_output(AIEnrichmentResponse)
            response = structured.invoke([HumanMessage(content=json.dumps(prompt))])
            parsed = response if isinstance(response, AIEnrichmentResponse) else AIEnrichmentResponse.model_validate(response)
        except Exception:
            return
        for key, suggestion in parsed.fields.items():
            if key not in target_fields or is_missing(suggestion.value):
                continue
            evidence_url = suggestion.source_url or (passages[0][1].url if passages else "")
            record[key] = str(suggestion.value).strip()
            traces[key] = FieldTrace(value=record[key], confidence_score=min(suggestion.confidence_score, 0.79), source=evidence_url or "OpenAI via LangChain", source_excerpt=evidence_context[:600], reasoning=suggestion.reasoning, status="evidenced_enrichment")


class ValidationAgent:
    """Checks schema adherence, URLs, confidence, and missing data flags."""

    URL_FIELDS = ["MFR URL", *[f"Ref URL {i}" for i in range(1, 6)]]

    def validate(self, record: Dict[str, Optional[str]], traces: Dict[str, FieldTrace]) -> Tuple[Dict[str, Optional[str]], ProductTraceability, ValidationReport]:
        errors: List[str] = []
        warnings: List[str] = []
        normalized = {header: record.get(header) for header in EXPECTED_HEADERS}
        try:
            model = ProductRecord(**normalized)
            normalized = model_to_delivery_dict(model)
        except Exception as exc:
            errors.append(f"Pydantic schema validation failed: {exc}")

        for field in self.URL_FIELDS:
            value = normalized.get(field)
            if value and not re.match(r"^https?://", value, flags=re.I):
                errors.append(f"{field} must be an http(s) URL when populated")
        missing_data: List[str] = []
        hitl_fields: List[str] = []
        complete_traces: Dict[str, FieldTrace] = {}
        for header in EXPECTED_HEADERS:
            trace = traces.get(header)
            value = normalized.get(header)
            if is_missing(value):
                # Missing fields are represented compactly in missing_data; only
                # populated/generated fields need field-level evidence.
                missing_data.append(header)
                continue
            if trace is None:
                trace = FieldTrace(value=str(value), confidence_score=0.75, source="pipeline", source_excerpt=str(value)[:240], reasoning="Value was preserved by the pipeline but has no source-level explanation.", status="observed")
            if trace.confidence_score < 0.8:
                hitl_fields.append(header)
            complete_traces[header] = trace
        if hitl_fields:
            warnings.append(f"{len(hitl_fields)} fields are below the 0.80 confidence threshold and require HITL review")
        return normalized, ProductTraceability(fields=complete_traces, missing_data=missing_data, hitl_fields=hitl_fields), ValidationReport(valid=not errors, errors=errors, warnings=warnings, hitl_required=bool(hitl_fields))


class Orchestrator:
    """Runs Ingestion -> Extraction -> Enrichment -> Validation."""

    def __init__(self, use_llm: Optional[bool] = None, research_sources: bool = False, research_limit: int = 5):
        self.ingestion = IngestionAgent()
        self.extraction = ExtractionAgent()
        self.use_llm = use_llm
        self.research_sources = research_sources
        self.research_limit = max(0, int(research_limit))
        self.source_research = SourceResearchAgent(use_llm=bool(use_llm))
        self.validation = ValidationAgent()

    def process(self, payload: Optional[bytes] = None, filename: str = "", text: Optional[str] = None, rows: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        ingestion = IngestionResult(rows=rows, raw_text="", filename="rows") if rows is not None else self.ingestion.ingest(payload=payload, filename=filename, text=text)
        kb = LocalKnowledgeBase(ingestion.rows)
        enrichment = EnrichmentAgent(kb, use_llm=self.use_llm)
        records: List[Dict[str, Optional[str]]] = []
        traceability: List[ProductTraceability] = []
        validation_reports: List[ValidationReport] = []
        research_results: List[Dict[str, Any]] = []
        for index, source in enumerate(ingestion.rows):
            record, traces = self.extraction.extract(source)
            record, traces = enrichment.enrich(record, traces, source)
            if self.research_sources and index < self.research_limit:
                record, traces, research_meta = self.source_research.enrich_from_sources(record, traces, source)
            else:
                research_meta = {"references": [], "documents": [], "evidence_count": 0, "warnings": ["Source research disabled or row outside research_limit"]}
            normalized, trace, report = self.validation.validate(record, traces)
            records.append(normalized)
            traceability.append(trace)
            validation_reports.append(report)
            research_results.append(research_meta)
        errors = [error for report in validation_reports for error in report.errors]
        warnings = [warning for report in validation_reports for warning in report.warnings]
        return {
            "records": records,
            "traceability": [trace.model_dump() for trace in traceability],
            "validation": ValidationReport(valid=not errors, errors=errors, warnings=warnings, hitl_required=any(report.hitl_required for report in validation_reports)).model_dump(),
            "delivery_headers": EXPECTED_HEADERS,
            "ingestion": {"filename": ingestion.filename, "row_count": len(ingestion.rows), "warnings": ingestion.warnings},
            "research": {
                "enabled": self.research_sources,
                "rows_researched": min(len(ingestion.rows), self.research_limit) if self.research_sources else 0,
                "evidence_items": sum(item.get("evidence_count", 0) for item in research_results),
                "rows": research_results,
            },
        }


def records_to_csv(records: List[Dict[str, Optional[str]]]) -> str:
    """Return the exact expected-output header order with no traceability columns."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=EXPECTED_HEADERS, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow({header: record.get(header) or "" for header in EXPECTED_HEADERS})
    return output.getvalue()
