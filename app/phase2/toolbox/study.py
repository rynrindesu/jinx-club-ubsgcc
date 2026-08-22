"""Token-capped retrieval from the Tool-box study materials."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import urljoin

import httpx


TOKEN_BUDGET = 900
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "at", "by", "did", "do", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "the", "to", "was",
        "what", "when", "where", "which", "who", "with",
    }
)


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


@dataclass(frozen=True)
class StudyDocument:
    title: str
    url: str
    text: str


class _StudyPageParser(HTMLParser):
    """Extract readable text and links without another HTML dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag == "title":
            self._in_title = True
        if tag in {"br", "div", "h1", "h2", "h3", "li", "p", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in {"div", "h1", "h2", "h3", "li", "p", "section"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)
        if self._in_title:
            self.title += data

    @property
    def text(self) -> str:
        return re.sub(r"[ \t]+", " ", "".join(self.parts)).strip()


def study_passages(question: str, study_materials_url: str) -> list[str]:
    """Retrieve the most relevant study excerpts within the exact token cap."""

    return select_passages(question, load_study_documents(study_materials_url))


@lru_cache(maxsize=4)
def load_study_documents(study_materials_url: str) -> tuple[StudyDocument, ...]:
    """Fetch the study-material index and every document it lists once."""

    response = httpx.get(study_materials_url, timeout=15.0, follow_redirects=True)
    response.raise_for_status()
    sources = _document_sources(response.text, study_materials_url)
    if not sources:
        raise ValueError("the study-material index did not list any documents")

    documents: list[StudyDocument] = []
    for source in sources:
        document_response = httpx.get(source, timeout=15.0, follow_redirects=True)
        document_response.raise_for_status()
        parser = _parse_html(document_response.text)
        text = parser.text or document_response.text.strip()
        if text:
            documents.append(
                StudyDocument(
                    title=parser.title.strip() or source.rsplit("/", 1)[-1],
                    url=source,
                    text=text,
                )
            )
    if not documents:
        raise ValueError("the study-material documents contained no text")
    return tuple(documents)


def select_passages(
    question: str,
    documents: tuple[StudyDocument, ...],
    encoder: Encoder | None = None,
) -> list[str]:
    """Select question-relevant excerpts, never exceeding 900 o200k tokens."""

    if not question.strip():
        raise ValueError("question must not be empty")
    if not documents:
        raise ValueError("no study documents are available")

    encoder = encoder or _o200k_encoder()
    terms = _query_terms(question)
    candidates: list[tuple[int, str]] = []
    for document in documents:
        for paragraph in _paragraphs(document.text):
            passage = f"{document.title}\n{paragraph}".strip()
            candidates.append((_relevance_score(passage, question, terms), passage))

    chosen: list[str] = []
    remaining = TOKEN_BUDGET
    seen: set[str] = set()
    for score, passage in sorted(candidates, key=lambda candidate: candidate[0], reverse=True):
        if score <= 0 or passage in seen or remaining == 0:
            continue
        seen.add(passage)
        tokens = encoder.encode(passage)
        if len(tokens) <= remaining:
            chosen.append(passage)
            remaining -= len(tokens)
        else:
            chosen.append(encoder.decode(tokens[:remaining]))
            remaining = 0

    if not chosen:
        # Return a bounded first passage even if a query shares no exact terms.
        tokens = encoder.encode(candidates[0][1])
        chosen.append(encoder.decode(tokens[:TOKEN_BUDGET]))
    return chosen


def _o200k_encoder() -> Encoder:
    try:
        import tiktoken
    except ImportError as error:  # pragma: no cover - dependency is declared.
        raise RuntimeError("tiktoken must be installed to enforce the token limit") from error
    return tiktoken.get_encoding("o200k_base")


def _document_sources(index_text: str, index_url: str) -> list[str]:
    """Read document URLs from either the supplied HTML page or JSON index."""

    try:
        payload = json.loads(index_text)
    except json.JSONDecodeError:
        payload = None

    raw_sources: list[str] = []
    if isinstance(payload, list):
        raw_sources = [_url_from_item(item) for item in payload]
    elif isinstance(payload, dict):
        records = payload.get("documents", payload.get("study_materials", []))
        if isinstance(records, list):
            raw_sources = [_url_from_item(item) for item in records]
    else:
        raw_sources = _parse_html(index_text).links

    sources: list[str] = []
    for source in raw_sources:
        if not source:
            continue
        absolute = urljoin(index_url, source)
        if absolute != index_url and absolute not in sources:
            sources.append(absolute)
    return sources


def _url_from_item(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("url", "href", "address"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return ""


def _parse_html(value: str) -> _StudyPageParser:
    parser = _StudyPageParser()
    parser.feed(value)
    parser.close()
    return parser


def _paragraphs(text: str) -> list[str]:
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
    ]
    return [paragraph for paragraph in paragraphs if paragraph]


def _query_terms(question: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", question.lower())
        if len(term) > 1 and term not in _STOP_WORDS
    }


def _relevance_score(passage: str, question: str, terms: set[str]) -> int:
    lowered = passage.lower()
    score = sum(lowered.count(term) for term in terms)
    phrase = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
    if phrase and phrase in lowered:
        score += len(terms) * 3
    return score
