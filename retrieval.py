from __future__ import annotations

import math
import re
from dataclasses import dataclass

from drive_loader import DriveDocument


@dataclass
class Chunk:
    chunk_id: str
    file_id: str
    file_name: str
    path: str
    web_view_link: str
    source_kind: str
    chunk_index: int
    chunk_total: int
    text: str


def build_chunks(
    documents: list[DriveDocument],
    chunk_chars: int = 1800,
    overlap_chars: int = 260,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for document in documents:
        document_chunks: list[Chunk] = []
        text = document.text.strip()
        if not text:
            continue

        start = 0
        chunk_index = 1

        while start < len(text):
            stop = min(len(text), start + chunk_chars)
            boundary = stop

            if stop < len(text):
                paragraph_boundary = text.rfind("\n", start, stop)
                sentence_boundary = text.rfind(". ", start, stop)
                boundary = max(paragraph_boundary, sentence_boundary, start + (chunk_chars // 2))
                boundary = min(boundary, stop)
                if boundary <= start:
                    boundary = stop

            chunk_text = text[start:boundary].strip()
            if chunk_text:
                document_chunks.append(
                    Chunk(
                        chunk_id=f"{document.file_id}:{chunk_index}",
                        file_id=document.file_id,
                        file_name=document.name,
                        path=document.path,
                        web_view_link=document.web_view_link,
                        source_kind=document.source_kind,
                        chunk_index=chunk_index,
                        chunk_total=0,
                        text=chunk_text,
                    )
                )
                chunk_index += 1

            if boundary >= len(text):
                break

            start = max(boundary - overlap_chars, start + 1)

        total = len(document_chunks)
        for chunk in document_chunks:
            chunk.chunk_total = total
            chunks.append(chunk)

    return chunks


def build_index(chunks: list[Chunk]) -> tuple[TfidfVectorizer | None, object | None]:
    if not chunks:
        return None, None

    corpus = [
        "\n".join(
            [
                chunk.file_name,
                chunk.path,
                chunk.source_kind,
                chunk.text,
                _normalize_search_text(
                    "\n".join(
                        [
                            chunk.file_name,
                            chunk.path,
                            chunk.source_kind,
                            chunk.text,
                        ]
                    )
                ),
            ]
        )
        for chunk in chunks
    ]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    except Exception:
        TfidfVectorizer = None
        cosine_similarity = None

    if TfidfVectorizer is not None and cosine_similarity is not None:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        matrix = vectorizer.fit_transform(corpus)
        return vectorizer, matrix

    # Fallback path for environments where sklearn cannot be imported.
    doc_terms: list[list[str]] = [_extract_search_terms(_normalize_search_text(text)) for text in corpus]
    doc_freq: dict[str, int] = {}
    for terms in doc_terms:
        for term in set(terms):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    total_docs = max(len(doc_terms), 1)
    idf = {term: math.log((1 + total_docs) / (1 + freq)) + 1 for term, freq in doc_freq.items()}
    matrix = [_build_weighted_vector(terms, idf) for terms in doc_terms]
    vectorizer = {
        "_fallback": True,
        "idf": idf,
    }
    return vectorizer, matrix


def retrieve(
    query: str,
    chunks: list[Chunk],
    vectorizer: TfidfVectorizer | None,
    matrix: object | None,
    limit: int = 8,
) -> list[tuple[Chunk, float]]:
    if not chunks or vectorizer is None or matrix is None:
        return []

    normalized_query = _normalize_search_text(query)
    if isinstance(vectorizer, dict) and vectorizer.get("_fallback"):
        idf = vectorizer.get("idf", {})
        query_terms = _extract_search_terms(normalized_query)
        query_vector = _build_weighted_vector(query_terms, idf)
        base_scores = [
            _cosine_similarity_sparse(query_vector, chunk_vector)
            for chunk_vector in matrix
        ]
    else:
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        query_vector = vectorizer.transform([f"{query}\n{normalized_query}"])
        base_scores = cosine_similarity(query_vector, matrix).flatten()

    query_lower = query.lower()
    normalized_query_lower = normalized_query.lower()
    terms = _extract_search_terms(normalized_query_lower)
    query_numbers = {term for term in terms if term.isdigit()}
    ranked: list[tuple[Chunk, float]] = []

    for index, chunk in enumerate(chunks):
        searchable = f"{chunk.file_name} {chunk.path} {chunk.text}".lower()
        normalized_searchable = _normalize_search_text(searchable)
        title_searchable = _normalize_search_text(f"{chunk.file_name} {chunk.path}")
        score = float(base_scores[index])
        title_hits = 0

        if query_lower in searchable or normalized_query_lower in normalized_searchable:
            score += 0.75

        if terms:
            title_hits = sum(term in title_searchable for term in terms)
            body_hits = sum(term in normalized_searchable for term in terms)
            score += title_hits * 0.22
            score += (body_hits / len(terms)) * 0.45

        if query_numbers:
            matched_numbers = sum(number in title_searchable for number in query_numbers)
            score += matched_numbers * 0.5

        if _looks_like_file_lookup(query) and title_hits:
            score += 0.55

        if chunk.chunk_index == 1 and title_hits:
            score += 0.12

        ranked.append((chunk, score))

    ranked.sort(key=lambda item: item[1], reverse=True)

    selected: list[tuple[Chunk, float]] = []
    for chunk, score in ranked[:limit]:
        selected.append((chunk, score))

    return selected


def _build_weighted_vector(terms: list[str], idf: dict[str, float]) -> dict[str, float]:
    if not terms:
        return {}

    counts: dict[str, int] = {}
    for term in terms:
        counts[term] = counts.get(term, 0) + 1

    total_terms = len(terms)
    weighted: dict[str, float] = {}
    for term, count in counts.items():
        weight = (count / total_terms) * idf.get(term, 1.0)
        weighted[term] = weight

    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm <= 0:
        return weighted

    return {term: value / norm for term, value in weighted.items()}


def _cosine_similarity_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    shared = set(a).intersection(b)
    if not shared:
        return 0.0
    return float(sum(a[term] * b[term] for term in shared))


def _normalize_search_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[_/\\-]+", " ", normalized)
    normalized = re.sub(r"\bsem\s*(\d+)\b", r"semester \1", normalized)
    normalized = re.sub(r"\b(\d+)\s*sem\b", r"semester \1", normalized)
    normalized = re.sub(r"\b(\d+)\s*(st|nd|rd|th)\b", r"semester \1 \1", normalized)
    normalized = re.sub(r"\bsemester\s*(\d+)\b", r"semester \1 \1", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_search_terms(text: str) -> list[str]:
    terms = []
    for token in re.findall(r"[a-z0-9]+", text):
        if token.isdigit() or len(token) > 2:
            terms.append(token)
    return list(dict.fromkeys(terms))


def _looks_like_file_lookup(query: str) -> bool:
    lowered = query.lower()
    lookup_words = {
        "file",
        "document",
        "doc",
        "pdf",
        "ppt",
        "pptx",
        "sheet",
        "syllabus",
        "semester",
        "title",
    }
    return any(word in lowered for word in lookup_words)


def build_context(retrieved: list[tuple[Chunk, float]], max_chars: int = 20000) -> str:
    if not retrieved:
        return ""

    sections = []
    total_chars = 0

    for rank, (chunk, score) in enumerate(retrieved, start=1):
        block = (
            f"[Source {rank}]\n"
            f"File: {chunk.file_name}\n"
            f"Path: {chunk.path}\n"
            f"Kind: {chunk.source_kind}\n"
            f"Chunk: {chunk.chunk_index}/{chunk.chunk_total}\n"
            f"Link: {chunk.web_view_link or 'Unavailable'}\n"
            f"Retrieval score: {score:.4f}\n"
            f"Content:\n{chunk.text}"
        )

        if total_chars + len(block) > max_chars:
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            sections.append(block[:remaining])
            break

        sections.append(block)
        total_chars += len(block)

    return "\n\n".join(sections)


def serialize_sources(retrieved: list[tuple[Chunk, float]]) -> list[dict[str, str]]:
    return [
        {
            "label": f"[Source {index}] {chunk.file_name}",
            "path": chunk.path,
            "kind": chunk.source_kind,
            "chunk": f"{chunk.chunk_index}/{chunk.chunk_total}",
            "link": chunk.web_view_link or "",
        }
        for index, (chunk, _score) in enumerate(retrieved, start=1)
    ]
