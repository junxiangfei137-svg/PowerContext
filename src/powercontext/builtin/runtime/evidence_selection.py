# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rank literal sentence spans of recalled Memory entries against one query.

The selection is deliberately offline: query terms are matched against the
candidate text with an IDF weight measured over the candidate set itself, so no
model call, annotation, or caller supplied answer influences the result. Spans
always reference the original characters; nothing here rewrites evidence.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

_SENTENCE_BOUNDARY = re.compile(r"[.!?]+[\"'”’)\]]*\s+|\n+")
_WORD = re.compile(r"[a-z]+|\d+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SPACE = re.compile(r"\s+")

_MAX_SENTENCE_CHARS = 800
_ADJACENCY_WEIGHT = 0.35

_STOPWORDS = frozenset(
    """
    a an the of and or in on at to for with by from as is are was were be been being this that these those it its
    their there here which who whom whose what when where how not no but if then than such into over under between
    among during about after before more most less least also may might can could should would will shall do does
    did done have has had we our you your they them her she his
    """.split()
)


@dataclass(frozen=True)
class EvidenceCandidate:
    """One query-relevant half-open character span of one candidate document."""

    document: int
    ordinal: int
    start: int
    end: int
    text: str
    score: float


def split_evidence_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Split one entry into sentence spans that keep their exact original offsets.

    The trailing whitespace of a sentence stays inside its span so that two
    adjacent selected sentences merge into one contiguous interval.
    """

    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.end()
        if end > start:
            _append_spans(text, start, end, spans)
        start = max(start, end)
    if start < len(text):
        _append_spans(text, start, len(text), spans)
    return tuple(spans)


def rank_evidence(query: str, documents: Sequence[str], /) -> tuple[EvidenceCandidate, ...]:
    """Return every sentence span ordered by how strongly it answers ``query``."""

    located: list[tuple[int, int, int, int, str]] = []
    for document, text in enumerate(documents):
        for ordinal, (start, end) in enumerate(split_evidence_spans(text)):
            body = text[start:end]
            if body.strip():
                located.append((document, ordinal, start, end, body))
    if not located:
        return ()
    scores = _score_spans(query, tuple(item[4] for item in located), tuple(item[0] for item in located))
    candidates = [
        EvidenceCandidate(document=document, ordinal=ordinal, start=start, end=end, text=body, score=score)
        for (document, ordinal, start, end, body), score in zip(located, scores, strict=True)
    ]
    ordered = sorted(candidates, key=lambda item: (-item.score, len(item.text), item.document, item.ordinal))
    seen: set[str] = set()
    unique: list[EvidenceCandidate] = []
    for candidate in ordered:
        key = _SPACE.sub(" ", candidate.text.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    if not any(candidate.score > 0 for candidate in unique):
        # Without a single lexical match, the opening sentence of each document is
        # still the most honest literal excerpt to offer.
        return tuple(
            candidate
            for candidate in sorted(candidates, key=lambda item: (item.document, item.ordinal))
            if candidate.ordinal == 0
        )
    return tuple(candidate for candidate in unique if candidate.score > 0)


def _append_spans(text: str, start: int, end: int, spans: list[tuple[int, int]]) -> None:
    while end - start > _MAX_SENTENCE_CHARS:
        window = text[start : start + _MAX_SENTENCE_CHARS]
        boundary = max(window.rfind("; "), window.rfind(", "))
        cut = start + boundary + 2 if boundary > 0 else start + _MAX_SENTENCE_CHARS
        spans.append((start, cut))
        start = cut
    if text[start:end].strip():
        spans.append((start, end))


def _score_spans(query: str, texts: Sequence[str], groups: Sequence[int]) -> tuple[float, ...]:
    terms = tuple(dict.fromkeys(_terms(query)))
    if not terms:
        return tuple(0.0 for _ in texts)
    query_terms = tuple(term for term in terms if term not in _STOPWORDS) or terms
    query_set = set(query_terms)
    tokenized = tuple(_terms(text) for text in texts)
    frequency: Counter[str] = Counter()
    for tokens in tokenized:
        frequency.update(set(tokens))
    total = len(texts)
    idf = {term: math.log((total + 1) / (frequency.get(term, 0) + 0.5)) for term in query_set}
    phrase = _SPACE.sub(" ", query.strip().lower())
    base: list[float] = []
    for text, tokens in zip(texts, tokenized, strict=True):
        present = set(tokens)
        score = sum(idf[term] for term in query_set & present)
        if score > 0:
            positions = {
                term: tuple(index for index, token in enumerate(tokens) if token == term) for term in query_set
            }
            for first, second in zip(query_terms, query_terms[1:], strict=False):
                if _adjacent(positions[first], positions[second]):
                    score += 0.5 * (idf[first] + idf[second])
            if phrase and phrase in _SPACE.sub(" ", text.lower()):
                score += 1.0
        base.append(score)
    return _smooth(base, groups)


def _smooth(scores: list[float], groups: Sequence[int]) -> tuple[float, ...]:
    """Let a sentence inherit part of its neighbours' strength: evidence spans sit together."""

    smoothed: list[float] = []
    for index, score in enumerate(scores):
        if score <= 0:
            smoothed.append(score)
            continue
        neighbours = [
            scores[other]
            for other in (index - 1, index + 1)
            if 0 <= other < len(scores) and groups[other] == groups[index]
        ]
        best = max(neighbours, default=0.0)
        smoothed.append(score + _ADJACENCY_WEIGHT * best)
    return tuple(smoothed)


def _adjacent(first: tuple[int, ...], second: tuple[int, ...]) -> bool:
    return any(left + 1 == right for left in first for right in second)


def _terms(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    terms = [_stem(token) for token in _WORD.findall(lowered)]
    cjk = _CJK.findall(lowered)
    terms.extend(cjk)
    terms.extend(cjk[index] + cjk[index + 1] for index in range(len(cjk) - 1))
    return tuple(terms)


def _stem(token: str) -> str:
    if not token.isascii() or not token.isalpha() or len(token) < 4:
        return token
    if token.endswith("ies"):
        return f"{token[:-3]}y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("ion") and len(token) > 6:
        return token[:-3]
    return token


__all__ = ["EvidenceCandidate", "rank_evidence", "split_evidence_spans"]
