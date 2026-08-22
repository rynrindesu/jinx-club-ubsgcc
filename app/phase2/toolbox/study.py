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
# The challenge permits 900 o200k_base tokens in total.  Three focused
# evidence passages preserve answer coverage without exceeding that ceiling.
MAX_PASSAGES = 3
MAX_PASSAGE_TOKENS = 300
MAX_WINDOW_SENTENCES = 3
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "at", "by", "did", "do", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "the", "to", "was",
        "what", "when", "where", "which", "who", "with",
    }
)
_SYNONYM_GROUPS = (
    frozenset({"fault", "failure", "malfunction", "breakdown", "defect"}),
    frozenset({"mechanical", "machinery", "machine", "equipment", "compressor", "engine"}),
    frozenset({"cold", "refrigeration", "refrigerated", "chilled", "freezer"}),
    frozenset({"threaten", "risk", "endanger", "jeopardize"}),
    frozenset({"cause", "because", "due", "reason", "root", "traced"}),
)
_STEM_EQUIVALENTS = {
    "caused": "cause",
    "causes": "cause",
    "causing": "cause",
    "stored": "store",
    "storage": "store",
    "stores": "store",
    "threatened": "threaten",
    "threatening": "threaten",
}
_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}\s+(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)|(?:january|february|march|april|may|"
    r"june|july|august|september|october|november|december)\s+\d{1,2})\b",
    re.IGNORECASE,
)


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


@dataclass(frozen=True)
class StudyDocument:
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class _SpanCandidate:
    score: int
    matched_terms: frozenset[str]
    source: tuple[str, str, int, int]
    passage: str


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
    phrases = _query_phrases(question)
    answer_signals = _answer_signals(question)
    candidates: list[_SpanCandidate] = []
    for document in documents:
        candidates.extend(_span_candidates(document, terms, phrases, answer_signals))
    if not candidates:
        raise ValueError("the study documents contained no usable sentences")

    chosen: list[str] = []
    remaining = TOKEN_BUDGET
    covered_terms: set[str] = set()
    selected_sources: list[tuple[str, str, int, int]] = []
    ranked = sorted(
        ((candidate, encoder.encode(candidate.passage)) for candidate in candidates),
        key=lambda item: (item[0].score - 0.02 * len(item[1]), item[0].score),
        reverse=True,
    )
    for candidate, tokens in ranked:
        if candidate.score <= 0 or remaining == 0 or len(chosen) == MAX_PASSAGES:
            continue
        if _overlaps_selected_window(candidate.source, selected_sources):
            continue
        if chosen and not candidate.matched_terms - covered_terms:
            continue

        passage_budget = min(remaining, MAX_PASSAGE_TOKENS)
        if len(tokens) <= passage_budget:
            chosen.append(candidate.passage)
            remaining -= len(tokens)
        else:
            chosen.append(encoder.decode(tokens[:passage_budget]))
            remaining -= passage_budget
        covered_terms.update(candidate.matched_terms)
        selected_sources.append(candidate.source)

    if not chosen:
        # Return one compact evidence span even if no words match exactly.
        tokens = encoder.encode(candidates[0].passage)
        passage_budget = min(TOKEN_BUDGET, MAX_PASSAGE_TOKENS)
        chosen.append(
            candidates[0].passage
            if len(tokens) <= passage_budget
            else encoder.decode(tokens[:passage_budget])
        )
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


def _span_candidates(
    document: StudyDocument,
    query_terms: set[str],
    query_phrases: set[str],
    answer_signals: set[str],
) -> list[_SpanCandidate]:
    """Generate compact, contiguous evidence spans around every sentence."""

    candidates: list[_SpanCandidate] = []
    for heading, section in _sections(document):
        sentences = _sentences(section)
        for start in range(len(sentences)):
            for end in range(
                start + 1,
                min(len(sentences), start + MAX_WINDOW_SENTENCES) + 1,
            ):
                evidence = " ".join(sentences[start:end])
                score, matched_terms = _span_score(
                    heading,
                    evidence,
                    query_terms,
                    query_phrases,
                    answer_signals,
                )
                answer_sentences = _answer_sentences(
                    heading,
                    sentences[start:end],
                    query_terms,
                    query_phrases,
                    answer_signals,
                )
                candidates.append(
                    _SpanCandidate(
                        score=score,
                        matched_terms=frozenset(matched_terms),
                        source=(document.url, heading, start, end),
                        passage=f"{heading}\n{' '.join(answer_sentences)}",
                    )
                )
    return candidates


def _sections(document: StudyDocument) -> list[tuple[str, str]]:
    """Split Markdown-style study material into heading-labelled sections."""

    sections: list[tuple[str, str]] = []
    heading = document.title
    lines: list[str] = []
    for line in document.text.splitlines():
        match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if match:
            body = "\n".join(lines).strip()
            if body:
                sections.append((heading, body))
            heading = match.group(1)
            lines = []
        else:
            lines.append(line)

    body = "\n".join(lines).strip()
    if body:
        sections.append((heading, body))
    return sections


def _sentences(text: str) -> list[str]:
    """Keep factual sentences intact while tolerating plain Markdown or HTML text."""

    normalised = re.sub(r"\s+", " ", text).strip()
    if not normalised:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalised)
        if sentence.strip()
    ]


def _overlaps_selected_window(
    source: tuple[str, str, int, int],
    selected_sources: list[tuple[str, str, int, int]],
) -> bool:
    url, heading, start, end = source
    return any(
        selected_url == url
        and selected_heading == heading
        and start < selected_end
        and selected_start < end
        for selected_url, selected_heading, selected_start, selected_end in selected_sources
    )


def _query_terms(question: str) -> set[str]:
    return {
        _normalise_term(term)
        for term in _tokens(question)
        if len(term) > 1 and term not in _STOP_WORDS
    }


def _query_phrases(question: str) -> set[str]:
    terms = [
        _normalise_term(term)
        for term in _tokens(question)
        if len(term) > 1 and term not in _STOP_WORDS
    ]
    return {f"{first} {second}" for first, second in zip(terms, terms[1:])}


def _answer_signals(question: str) -> set[str]:
    lowered = question.lower()
    signals: set[str] = set()
    if "when" in lowered or "date" in lowered or "what day" in lowered:
        signals.add("date")
    if "how many" in lowered or "how much" in lowered or "number of" in lowered:
        signals.add("number")
    if "why" in lowered or any(
        phrase in lowered for phrase in ("what caused", "reason for", "due to")
    ):
        signals.add("cause")
    return signals


def _answer_sentences(
    heading: str,
    sentences: list[str],
    query_terms: set[str],
    query_phrases: set[str],
    answer_signals: set[str],
) -> list[str]:
    """Keep only sentences that directly carry the requested answer type."""

    scored = [
        (
            _span_score(heading, sentence, query_terms, query_phrases, answer_signals),
            sentence,
        )
        for sentence in sentences
    ]
    intent_sentences = [
        sentence
        for (_, sentence) in scored
        if _matches_answer_signal(sentence, answer_signals)
    ]
    if intent_sentences:
        return intent_sentences

    # For questions without a date/number/cause cue, retain only the sentence
    # with the strongest direct evidence instead of emitting all span context.
    return [max(scored, key=lambda item: item[0][0])[1]]


def _matches_answer_signal(sentence: str, answer_signals: set[str]) -> bool:
    if "date" in answer_signals and _DATE_PATTERN.search(sentence):
        return True
    if "number" in answer_signals and re.search(r"\b\d+(?:\.\d+)?\b", sentence):
        return True
    if "cause" in answer_signals:
        sentence_terms = {_normalise_term(term) for term in _tokens(sentence)}
        return bool(sentence_terms & _synonyms_for("cause"))
    return False


def _span_score(
    heading: str,
    evidence: str,
    query_terms: set[str],
    query_phrases: set[str],
    answer_signals: set[str],
) -> tuple[int, set[str]]:
    evidence_terms = {_normalise_term(term) for term in _tokens(f"{heading} {evidence}")}
    exact_matches = evidence_terms & query_terms
    synonym_matches = {
        term
        for term in query_terms - exact_matches
        if evidence_terms & _synonyms_for(term)
    }
    score = len(exact_matches) * 10 + len(synonym_matches) * 4

    # Reward adjacent query phrases such as "cold store" without requiring the
    # source to use the same hyphenation as the question.
    normalised_evidence = " ".join(_normalise_term(term) for term in _tokens(evidence))
    for phrase in query_phrases:
        if phrase in normalised_evidence:
            score += 4
    if "date" in answer_signals and _DATE_PATTERN.search(evidence):
        score += 6
    if "number" in answer_signals and re.search(r"\b\d+(?:\.\d+)?\b", evidence):
        score += 4
    return score, exact_matches | synonym_matches


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("-", " "))


def _normalise_term(term: str) -> str:
    return _STEM_EQUIVALENTS.get(term, term)


def _synonyms_for(term: str) -> frozenset[str]:
    for group in _SYNONYM_GROUPS:
        if term in group:
            return group
    return frozenset({term})
