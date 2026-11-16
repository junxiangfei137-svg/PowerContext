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
from collections import Counter
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
    fy fiscal year years question answer respond response calculate calculated define defined assume
    assuming perspective analyst investment give relying rely detail details show shown state stated
    outline outlined directly explicitly else much many according provide provided following
    company improving improve profile useful metric like explain able reasonably healthy measure
    historically consistent consistently fluctuating fluctuate roughly relevant represent
    """.split()
)
_MAX_TERM_QUERIES = 24
_MAX_EXPANSION_QUERIES = 24
_MAX_EXPANSION_TERMS = 64
_CANDIDATE_LIMIT = 1000

_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "geography", "geographies", "geographic", "geographical", "region", "regions",
        "country", "countries", "international", "worldwide", "global", "europe", "asia", "africa",
    ),
    (
        "operate", "operates", "operating", "operation", "operations", "business", "businesses",
        "segment", "segments",
    ),
    (
        "legal", "litigation", "litigations", "lawsuit", "lawsuits", "proceeding", "proceedings",
        "claim", "claims", "dispute", "disputes", "regulatory",
    ),
    ("battle", "battles", "litigation", "lawsuit", "proceedings", "dispute"),
    (
        "restructuring", "restructurings", "impairment", "impairments", "charges", "severance",
        "productivity",
    ),
    ("ebitda", "ebit", "depreciation", "amortization", "amortisation"),
    ("capex", "capital", "expenditure", "expenditures", "property", "plant", "equipment"),
    ("revenue", "revenues", "sales", "sale"),
    ("margin", "margins", "profit", "profits", "earnings"),
    ("debt", "debts", "borrowings", "borrowing", "notes", "payable"),
    ("dividend", "dividends"),
    ("repurchase", "repurchases", "buyback", "buybacks"),
    ("employee", "employees", "workforce", "headcount", "personnel"),
    ("competition", "competitive", "competitor", "competitors"),
    ("risk", "risks", "uncertainties"),
    ("tax", "taxes", "taxation"),
    ("lease", "leases", "rental"),
    ("goodwill", "impairment"),
    ("pension", "pensions", "retirement"),
    ("acquisition", "acquisitions", "acquired", "merger", "mergers"),
    ("inventory", "inventories"),
    ("receivable", "receivables"),
    ("payable", "payables"),
    ("shareholder", "shareholders", "stockholder", "stockholders", "equity"),
    ("retain", "retains", "retained", "retaining", "retention"),
    ("customer", "customers", "client", "clients", "consumer", "consumers"),
    ("cybersecurity", "cyber", "security", "breach"),
    ("supply", "supplies", "supplier", "suppliers", "logistics", "distribution"),
    ("climate", "environmental", "emissions", "sustainability", "carbon"),
)
_SYNONYM_INDEX: dict[str, tuple[str, ...]] = {}
for _group in _SYNONYM_GROUPS:
    for _member in _group:
        _SYNONYM_INDEX[_member] = tuple(dict.fromkeys((*_SYNONYM_INDEX.get(_member, ()), *_group)))

_ANCHOR_PHRASES: dict[tuple[str, str], tuple[str, ...]] = {
    ("income", "statement"): ("statement of income", "consolidated statement of income", "statement of operations"),
    ("statement", "income"): ("statement of income", "consolidated statement of income", "statement of operations"),
    ("cash", "flow"): ("statement of cash flows", "consolidated statement of cash flows", "cash flows from operating activities"),
    ("cash", "flows"): ("statement of cash flows", "consolidated statement of cash flows", "cash flows from operating activities"),
    ("flow", "statement"): ("statement of cash flows", "consolidated statement of cash flows"),
    ("flows", "statement"): ("statement of cash flows", "consolidated statement of cash flows"),
    ("balance", "sheet"): ("consolidated balance sheet", "balance sheet"),
    ("legal", "proceedings"): ("legal proceedings", "item 3 legal proceedings"),
    ("proceedings", "legal"): ("legal proceedings", "item 3 legal proceedings"),
    ("legal", "battles"): ("legal proceedings", "litigation", "legal matters"),
    ("legal", "matters"): ("legal proceedings", "legal matters"),
    ("legal", "battle"): ("legal proceedings", "litigation"),
    ("restructuring", "costs"): ("restructuring and impairment", "restructuring charges", "restructuring and impairment charges"),
    ("restructuring", "charges"): ("restructuring and impairment", "restructuring charges", "restructuring and impairment charges"),
    ("risk", "factors"): ("risk factors",),
    ("capital", "expenditures"): ("capital expenditures", "purchases of property plant and equipment"),
    ("capital", "expenditure"): ("capital expenditures", "purchases of property plant and equipment"),
    ("property", "plant"): ("property plant and equipment",),
    ("operating", "income"): ("operating income", "operating profit"),
    ("gross", "margin"): ("gross margin", "gross profit"),
    ("net", "income"): ("net income", "net earnings"),
    ("free", "cash"): ("free cash flow",),
    ("interest", "expense"): ("interest expense",),
    ("income", "taxes"): ("income taxes", "effective tax rate"),
    ("card", "members"): ("card member", "card member retention", "card member spending"),
    ("card", "member"): ("card member", "card member retention", "card member spending"),
    ("primary", "customers"): (
        "commercial airlines",
        "u.s. government",
        "government contracts",
        "limited number of",
        "portion of our revenue",
        "portion of our revenues",
        "of our revenues",
        "customer accounted for",
    ),
    ("customers", "primary"): (
        "commercial airlines",
        "u.s. government",
        "government contracts",
        "limited number of",
        "portion of our revenue",
        "portion of our revenues",
        "of our revenues",
        "customer accounted for",
    ),
    ("customer", "concentration"): (
        "customer accounted for",
        "one customer",
        "significant customer",
        "major customer",
        "of our net revenue",
        "of our revenue",
    ),
    ("customer", "concentrations"): (
        "customer accounted for",
        "one customer",
        "significant customer",
        "major customer",
        "of our net revenue",
        "of our revenue",
    ),
    ("quick", "ratio"): ("current assets", "current liabilities", "total current assets", "total current liabilities"),
}
_STRONG_ANCHORS = frozenset(
    {
        "statement of income",
        "consolidated statement of income",
        "statement of operations",
        "statement of cash flows",
        "consolidated statement of cash flows",
        "cash flows from operating activities",
        "legal proceedings",
        "item 3 legal proceedings",
        "restructuring and impairment",
        "risk factors",
        "commercial airlines",
        "u.s. government",
        "government contracts",
    }
)

_STATEMENT_QUERIES = (
    "consolidated statements of income",
    "consolidated statement of income",
    "consolidated statements of operations",
    "consolidated statements of earnings",
    "consolidated balance sheets",
    "consolidated balance sheet",
    "consolidated statements of cash flows",
    "consolidated statement of cash flows",
    "summary of business segment data",
    "segment information",
)
_STATEMENT_HEADINGS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "income",
        re.compile(r"\b(consolidated )?(statements? of (income|operations|earnings)|statements? of income)\b"),
    ),
    ("balance", re.compile(r"\b(consolidated )?balance sheets?\b")),
    ("cash", re.compile(r"\b(consolidated )?statements? of cash flows?\b")),
    ("segment", re.compile(r"\b(business segment data|segment (data|information))\b")),
)
_KIND_CUES: dict[str, frozenset[str]] = {
    "income": frozenset(
        {
            "income", "revenue", "revenues", "sales", "sale", "margin", "margins", "profit",
            "profits", "earnings", "cost", "costs", "expense", "expenses", "operating", "operation",
            "operations",
        }
    ),
    "balance": frozenset(
        {
            "liability", "liabilities", "asset", "assets", "liquidity", "ratio", "balance", "sheet",
            "receivable", "receivables", "inventory", "inventories", "payable", "payables", "debt",
            "borrowings", "equity", "current",
        }
    ),
    "cash": frozenset(
        {
            "cash", "flow", "flows", "dividend", "dividends", "repurchase", "repurchases", "buyback",
            "buybacks", "capex", "expenditure", "expenditures", "depreciation", "amortization",
        }
    ),
    "segment": frozenset(
        {
            "segment", "segments", "product", "products", "service", "services", "category",
            "categories",
        }
    ),
}
_METRIC_CUES = frozenset().union(*_KIND_CUES.values()) | frozenset(
    {
        "percent", "percentage", "largest", "highest", "lowest", "total", "amount", "compare",
        "compared", "change", "growth", "increase", "decrease", "return", "turnover", "average",
    }
)


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
    terms = [
        token
        for token in _TOKEN_RE.findall(question.lower())
        if len(token) > 1 and token not in _STOPWORDS
    ]
    return tuple(dict.fromkeys(terms))


def _morph_variants(term: str) -> tuple[str, ...]:
    if len(term) < 5:
        return ()
    variants: list[str] = []
    if term.endswith("ies"):
        variants.append(term[:-3] + "y")
    elif term.endswith("es"):
        variants.append(term[:-2])
    elif term.endswith("s"):
        variants.append(term[:-1])
    if term.endswith("ing"):
        variants.extend((term[:-3], term[:-3] + "e"))
    elif term.endswith("ed"):
        variants.extend((term[:-2], term[:-1]))
    return tuple(variant for variant in dict.fromkeys(variants) if len(variant) > 2)


def _expansion_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    originals = set(terms)
    expanded: list[str] = []
    for term in terms:
        for candidate in (*_SYNONYM_INDEX.get(term, ()), *_morph_variants(term)):
            if candidate not in originals and candidate not in _STOPWORDS:
                expanded.append(candidate)
    return tuple(dict.fromkeys(expanded))[:_MAX_EXPANSION_TERMS]


def _anchor_phrases(terms: tuple[str, ...]) -> tuple[str, ...]:
    phrases: list[str] = []
    for left, right in zip(terms, terms[1:]):
        for pair in ((left, right), (right, left)):
            phrases.extend(_ANCHOR_PHRASES.get(pair, ()))
    return tuple(dict.fromkeys(phrases))


def _flat_text(text: str) -> str:
    body = text.split("\n", 1)[1] if "\n" in text else text
    return " ".join(_TOKEN_RE.findall(body.lower()))


def _contains_phrase(flat: str, phrase: str) -> bool:
    return f" {phrase} " in f" {flat} "


def _bm25(term_frequency: int, length: int, average_length: float, idf: float) -> float:
    if term_frequency < 1:
        return 0.0
    k1 = 1.2
    b = 0.75
    norm = k1 * (1.0 - b + b * (length / max(average_length, 1.0)))
    return idf * (term_frequency * (k1 + 1.0)) / (term_frequency + norm)


def _term_frequency(term: str, counts: Counter[str]) -> int:
    frequency = counts.get(term, 0)
    if frequency:
        return frequency
    if len(term) >= 5:
        prefix = term[:5]
        return sum(value for token, value in counts.items() if token.startswith(prefix))
    return 0


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple:
    """Return exact page citations ranked for one natural-language question."""

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime import SearchMemoryRequest

    if limit < 1:
        return ()
    terms = _query_terms(question)
    if not terms:
        terms = tuple(token for token in _TOKEN_RE.findall(question.lower())) or (question.strip().lower(),)
    expansions = _expansion_terms(terms)
    anchors = _anchor_phrases(terms)
    question_kinds = frozenset(
        kind for kind, cues in _KIND_CUES.items() if any(term in cues for term in terms)
    )
    quantitative = question_kinds or any(term in _METRIC_CUES for term in terms)

    queries: list[str] = [" ".join(terms[:_MAX_TERM_QUERIES])]
    queries.extend(terms[:_MAX_TERM_QUERIES])
    queries.extend(expansions[:_MAX_EXPANSION_QUERIES])
    if expansions:
        queries.append(" ".join(expansions))
    queries.extend(anchors)
    queries.extend(_STATEMENT_QUERIES)
    hits: dict[str, object] = {}
    seen_queries: set[str] = set()
    for query in queries:
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=_CANDIDATE_LIMIT))
        for hit in page.hits:
            hits.setdefault(hit.entry_id, hit)

    if not hits:
        return ()

    flats = {entry_id: _flat_text(hit.text) for entry_id, hit in hits.items()}
    counts = {entry_id: Counter(flat.split()) for entry_id, flat in flats.items()}
    lengths = {entry_id: len(flat.split()) for entry_id, flat in flats.items()}
    digits = {
        entry_id: sum(1 for token in flat.split() if token.isdigit()) for entry_id, flat in flats.items()
    }
    kinds = {
        entry_id: frozenset(
            kind
            for kind, pattern in _STATEMENT_HEADINGS
            if pattern.search(" ".join(flat.split()[:45]))
        )
        for entry_id, flat in flats.items()
    }
    average_length = sum(lengths.values()) / max(len(hits), 1)
    total = max(len(hits), 1)
    vocabulary = tuple(dict.fromkeys((*terms, *expansions)))
    document_frequency = {
        term: sum(1 for page_counts in counts.values() if term in page_counts) for term in vocabulary
    }
    idf = {
        term: math.log(1.0 + (total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
        for term in vocabulary
    }
    total_idf = sum(idf[term] for term in terms) or 1.0

    def score(entry_id: str) -> float:
        page_counts = counts[entry_id]
        length = lengths[entry_id]
        flat = flats[entry_id]
        value = 0.0
        matched_idfs: list[float] = []
        for term in terms:
            frequency = _term_frequency(term, page_counts)
            if not frequency:
                continue
            matched_idfs.append(idf[term])
            value += _bm25(frequency, length, average_length, idf[term])
        coverage = sum(matched_idfs) / total_idf
        value += 4.0 * coverage
        matched_idfs.sort(reverse=True)
        value += 1.2 * sum(matched_idfs[:2])
        expansion_value = 0.0
        for term in expansions:
            frequency = _term_frequency(term, page_counts)
            if frequency:
                expansion_value += _bm25(frequency, length, average_length, idf[term])
        value += min(2.0, 0.25 * expansion_value)
        pair_hits = sum(
            1
            for left, right in zip(terms, terms[1:])
            if _contains_phrase(flat, f"{left} {right}") or _contains_phrase(flat, f"{right} {left}")
        )
        value += min(3.0, 1.5 * pair_hits)
        anchor_hits = [phrase for phrase in anchors if _contains_phrase(flat, phrase)]
        value += sum(6.0 if phrase in _STRONG_ANCHORS else 4.0 for phrase in anchor_hits)
        heading = " ".join(flat.split()[:18])
        if any(_contains_phrase(heading, phrase) for phrase in anchors):
            value += 1.0
        page_kinds = kinds[entry_id]
        if page_kinds:
            value += 2.0
            if question_kinds and page_kinds & question_kinds:
                value += 13.0
        if quantitative:
            digit_share = digits[entry_id] / max(length, 1)
            if digit_share >= 0.25:
                value += 3.0
            elif digit_share >= 0.15:
                value += 1.5
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
