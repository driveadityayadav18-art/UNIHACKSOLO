"""Source-grounded research utilities for the UniHack product pipeline.

All fetched web content is treated as data only. The module extracts readable
text and evidence snippets; it never executes or follows instructions found in
remote documents.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None


@dataclass
class SourceReference:
    url: str
    title: str = ""
    discovery_query: str = ""
    rank: int = 0


@dataclass
class SourceDocument:
    url: str
    title: str
    text: str
    content_type: str = "text/html"
    retrieved_at: float = 0.0
    error: str = ""


@dataclass
class Evidence:
    field: str
    value: str
    source_url: str
    source_title: str
    excerpt: str
    confidence_score: float
    reasoning: str


class ResearchCache:
    """Small JSON cache to prevent repeated public fetches during a demo."""

    def __init__(self, path: str = ".cache/source_cache.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except Exception:
            self.data = {}

    def get(self, url: str) -> Optional[SourceDocument]:
        value = self.data.get(self._key(url))
        if not value:
            return None
        return SourceDocument(**value)

    def put(self, document: SourceDocument) -> None:
        self.data[self._key(document.url)] = asdict(document)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _key(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()


class SourceDiscoveryAgent:
    """Discovers public candidate pages without assuming a manufacturer URL."""

    SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"

    def __init__(self, timeout: int = 8, session: Optional[requests.Session] = None):
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "UniHack-Product-Intelligence/1.0"})

    def discover(self, row: Dict[str, str], max_results: int = 3) -> Tuple[List[SourceReference], List[str]]:
        part = str(row.get("Mfg_Part_Num", "")).strip()
        desc = str(row.get("Part_Desc", "")).strip()
        manufacturer = str(row.get("Part_Manuf", "")).strip()
        query = " ".join(item for item in (part, desc, manufacturer) if item and item not in {"-", "--"})
        if not query:
            return [], ["No searchable part number, description, or manufacturer was provided"]
        try:
            response = self.session.get(self.SEARCH_URL.format(query=quote_plus(query)), timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            return [], [f"Source discovery unavailable: {exc.__class__.__name__}"]

        soup = BeautifulSoup(response.text, "html.parser")
        references: List[SourceReference] = []
        seen: set[str] = set()
        for rank, anchor in enumerate(soup.select("a.result__a"), start=1):
            href = anchor.get("href", "").strip()
            url = self._unwrap(href)
            if not url or not url.startswith(("http://", "https://")) or url in seen:
                continue
            if self._is_search_or_social(url):
                continue
            seen.add(url)
            references.append(SourceReference(url=url, title=" ".join(anchor.get_text(" ", strip=True).split()), discovery_query=query, rank=rank))
            if len(references) >= max_results:
                break
        return references, [] if references else ["No public source candidates were found"]

    @staticmethod
    def _unwrap(url: str) -> str:
        parsed = urlparse(url)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            return unquote(parse_qs(parsed.query).get("uddg", [url])[0])
        return url

    @staticmethod
    def _is_search_or_social(url: str) -> bool:
        domain = urlparse(url).netloc.lower()
        return any(blocked in domain for blocked in ("duckduckgo.com", "google.com/search", "bing.com/search", "facebook.com", "instagram.com"))


class DocumentIngestionAgent:
    """Fetches and cleans HTML/PDF text, preserving URL-level provenance."""

    def __init__(self, cache: Optional[ResearchCache] = None, timeout: int = 10, session: Optional[requests.Session] = None):
        self.cache = cache or ResearchCache()
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "UniHack-Product-Intelligence/1.0"})

    def fetch(self, reference: SourceReference) -> SourceDocument:
        cached = self.cache.get(reference.url)
        if cached and not cached.error:
            return cached
        try:
            response = self.session.get(reference.url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" in content_type or reference.url.lower().split("?")[0].endswith(".pdf"):
                document = self._pdf(reference.url, reference.title, response.content, content_type)
            else:
                document = self._html(response.url, reference.title, response.text, content_type)
        except requests.RequestException as exc:
            document = SourceDocument(url=reference.url, title=reference.title, text="", content_type="", retrieved_at=time.time(), error=f"Fetch failed: {exc.__class__.__name__}")
        except Exception as exc:
            document = SourceDocument(url=reference.url, title=reference.title, text="", content_type="", retrieved_at=time.time(), error=f"Document parsing failed: {exc.__class__.__name__}")
        self.cache.put(document)
        return document

    @staticmethod
    def _html(url: str, title: str, html: str, content_type: str) -> SourceDocument:
        soup = BeautifulSoup(html, "html.parser")
        for node in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
            node.decompose()
        page_title = title or (soup.title.get_text(" ", strip=True) if soup.title else "")
        text = " ".join(soup.get_text(" ", strip=True).split())
        return SourceDocument(url=url, title=page_title[:300], text=text[:500_000], content_type=content_type or "text/html", retrieved_at=time.time())

    @staticmethod
    def _pdf(url: str, title: str, payload: bytes, content_type: str) -> SourceDocument:
        if PdfReader is None:
            return SourceDocument(url=url, title=title, text="", content_type=content_type or "application/pdf", retrieved_at=time.time(), error="pypdf is not installed")
        reader = PdfReader(io.BytesIO(payload))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(f"[Page {index}] {page}" for index, page in enumerate(pages, start=1))
        return SourceDocument(url=url, title=title or "PDF document", text=text[:500_000], content_type=content_type or "application/pdf", retrieved_at=time.time())


class LocalDocumentRAG:
    """Retrieves evidence chunks using TF-IDF cosine similarity, with a fallback."""

    def __init__(self, documents: Optional[Iterable[SourceDocument]] = None):
        self.chunks: List[Tuple[SourceDocument, str]] = []
        for document in documents or []:
            self.add_document(document)
        self.vectorizer = None
        self.matrix = None
        self._fit()

    def add_document(self, document: SourceDocument, chunk_chars: int = 900) -> None:
        if document.error or not document.text.strip():
            return
        text = re.sub(r"\s+", " ", document.text).strip()
        for start in range(0, len(text), chunk_chars):
            chunk = text[start : start + chunk_chars]
            if len(chunk) >= 40:
                self.chunks.append((document, chunk))
        self._fit()

    def _fit(self) -> None:
        if self.chunks and TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30_000)
            self.matrix = self.vectorizer.fit_transform([chunk for _, chunk in self.chunks])

    def search(self, query: str, k: int = 5) -> List[Tuple[float, SourceDocument, str]]:
        if not self.chunks:
            return []
        if self.vectorizer is not None and self.matrix is not None and cosine_similarity is not None:
            scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
            indices = scores.argsort()[::-1][:k]
            return [(float(scores[index]), self.chunks[index][0], self.chunks[index][1]) for index in indices if scores[index] > 0]
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for document, chunk in self.chunks:
            tokens = set(re.findall(r"[a-z0-9]+", chunk.lower()))
            score = len(query_tokens & tokens) / max(1, len(query_tokens | tokens))
            scored.append((score, document, chunk))
        return sorted(scored, key=lambda item: item[0], reverse=True)[:k]


def find_evidence(rag: LocalDocumentRAG, query: str, field: str, value: str, min_score: float = 0.06) -> Optional[Evidence]:
    results = rag.search(query, k=5)
    if not results:
        return None
    score, document, excerpt = results[0]
    if score < min_score:
        return None
    return Evidence(field=field, value=value, source_url=document.url, source_title=document.title, excerpt=excerpt[:600], confidence_score=min(0.95, 0.55 + score), reasoning=f"Matched the extracted value against a retrieved source passage with TF-IDF relevance {score:.2f}.")
