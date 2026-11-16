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

"""Parse a complete PDF into page-level Sources and retrieve evidence pages."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections.abc import Iterable
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    TextEvidence,
)
from powercontext.sources.models import Source, SourceMaterialization

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_DEFINITION_VERSION = "1"

_TOKEN_RE = re.compile(r"[a-z]+|\d+")
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below
    between both but by can did do does doing down during each few for from further had has have having
    he her here hers herself him himself his how i if in into is it its itself just me more most my myself
    no nor not now of off on once only or other ought our ours ourselves out over own same she should so
    some such than that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves per also may might must shall upon within without using use used based
    """.split()
)
_MAX_QUERY_TERMS = 8
_CANDIDATE_LIMIT = 200


class PDFDocumentInput(BaseModel):
    """One complete offline PDF document."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """Adapter-native value for one physical PDF page."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured Source holding one physical page of a PDF revision."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-blank string")  # noqa: TRY003
    source_id = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"pdf page source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return source_id


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page, returning no partial result on failure."""

    if not value.document_id.strip():
        raise ValueError("document_id must be a non-blank string")  # noqa: TRY003
    if not value.content:
        raise ValueError("PDF content must not be empty")  # noqa: TRY003

    digest = hashlib.sha256(value.content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(value.content), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a password to read")  # noqa: TRY003
        page_count = len(reader.pages)
        pages = [
            _page_input(value.document_id, digest, page_number, page)
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    except ValueError:
        raise
    except Exception as error:  # pypdf raises several internal exception types for corrupt input
        raise ValueError("PDF document could not be parsed completely") from error

    if len(pages) != page_count:
        raise ValueError("PDF document could not be parsed completely")  # noqa: TRY003
    return tuple(pages)


def _page_input(document_id: str, digest: str, page_number: int, page: object) -> PDFPageInput:
    try:
        extracted = page.extract_text()  # type: ignore[attr-defined]
    except Exception as error:
        raise ValueError(f"PDF page {page_number} could not be parsed") from error  # noqa: TRY003
    text = extracted if isinstance(extracted, str) else ""
    return PDFPageInput(
        document_id=document_id,
        document_sha256=digest,
        page_number=page_number,
        text=text,
    )


class PDFPageSourceAdapter:
    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    version=PDF_PAGE_DEFINITION_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)


def _query_terms(question: str) -> tuple[str, ...]:
    terms = [token for token in _TOKEN_RE.findall(question.lower()) if token not in _STOPWORDS]
    seen: set[str] = set()
    ordered: list[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            ordered.append(term)
    return tuple(ordered)


def _search_keywords(terms: Iterable[str]) -> str:
    return " ".join(terms)


def _page_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _match_weight(term: str, counts: dict[str, int]) -> float:
    if term in counts:
        return float(counts[term])
    if len(term) >= 5:
        prefix = term[:5]
        return float(sum(value for token, value in counts.items() if token.startswith(prefix))) * 0.6
    return 0.0


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple:
    """Return exact page citations ranked for one natural-language question."""

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime import SearchMemoryRequest

    if limit < 1:
        return ()
    terms = _query_terms(question)
    if not terms:
        terms = tuple(token for token in _TOKEN_RE.findall(question.lower())) or (question.strip().lower(),)

    hits: dict[str, object] = {}
    queries: list[str] = [_search_keywords(terms[:_MAX_QUERY_TERMS])]
    queries.extend(terms[:5])
    if len(terms) > 1:
        queries.append(_search_keywords(terms[:2]))
    seen_queries: set[str] = set()
    for query in queries:
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=_CANDIDATE_LIMIT))
        for hit in page.hits:
            current = hits.get(hit.entry_id)
            if current is None or hit.score > current.score:
                hits[hit.entry_id] = hit

    if not hits:
        return ()

    texts = {entry_id: hit.text.lower() for entry_id, hit in hits.items()}
    token_counts = {entry_id: _page_tokens(text) for entry_id, text in texts.items()}
    document_frequency = {term: 0 for term in terms}
    for entry_id, tokens in token_counts.items():
        token_set = set(tokens)
        for term in terms:
            if term in token_set or (len(term) >= 5 and any(token.startswith(term[:5]) for token in token_set)):
                document_frequency[term] += 1
    total = max(len(hits), 1)

    def score(entry_id: str) -> float:
        counts: dict[str, int] = {}
        for token in token_counts[entry_id]:
            counts[token] = counts.get(token, 0) + 1
        value = 0.0
        matched = 0
        for term in terms:
            weight = _match_weight(term, counts)
            if not weight:
                continue
            matched += 1
            idf = math.log(1.0 + total / (1.0 + document_frequency[term]))
            saturation = weight / (weight + 1.2)
            number_boost = 1.6 if term.isdigit() else 1.0
            value += idf * saturation * number_boost
        value += 0.35 * matched / max(len(terms), 1)
        phrase = _search_keywords(terms[:3])
        if phrase and phrase in texts[entry_id]:
            value += 0.5
        return value

    ranked = sorted(
        hits.values(),
        key=lambda hit: (score(hit.entry_id), hit.score),
        reverse=True,
    )[:limit]
    return tuple(
        MemoryCitation(memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id)
        for hit in ranked
    )


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
