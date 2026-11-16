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

"""CSV row Sources: one captured Source per offline table record."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_DEFINITION_VERSION = "1"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document to parse into row Sources."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """Adapter-native value for one CSV record."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured Source holding one row and its table identity."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def _validate_document_id(document_id: str) -> None:
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the stable ``csv:<document>:<key>`` identity for one row."""

    _validate_document_id(document_id)
    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary_key_value must be a non-blank string")  # noqa: TRY003
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(  # noqa: TRY003
            f"csv row source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters"
        )
    return source_id


def _validate_header(header: list[str]) -> None:
    if not header:
        raise ValueError("CSV header must not be empty")  # noqa: TRY003
    for column in header:
        if not column.strip():
            raise ValueError("CSV header fields must not be empty or blank")  # noqa: TRY003
    seen: set[str] = set()
    for column in header:
        if column in seen:
            raise ValueError(f"duplicate CSV header field: {column!r}")  # noqa: TRY003
        seen.add(column)


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and fully validate one CSV document before returning any row."""

    _validate_document_id(value.document_id)
    if not value.primary_key:
        raise ValueError("primary_key must not be empty")  # noqa: TRY003

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error  # noqa: TRY003

    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"malformed CSV content: {error}") from error  # noqa: TRY003

    if not records:
        raise ValueError("CSV document must contain a header record")  # noqa: TRY003

    header, *data = records
    _validate_header(header)
    if value.primary_key not in header:
        raise ValueError(f"primary key column not found in CSV header: {value.primary_key!r}")  # noqa: TRY003
    primary_key_index = header.index(value.primary_key)
    columns = tuple(header)

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for record in data:
        if not record:
            continue
        if len(record) != len(columns):
            raise ValueError(  # noqa: TRY003
                f"CSV record width {len(record)} does not match header width {len(columns)}"
            )
        primary_key_value = record[primary_key_index]
        if not primary_key_value.strip():
            raise ValueError("CSV primary key value must not be empty or blank")  # noqa: TRY003
        if primary_key_value in seen_keys:
            raise ValueError(f"duplicate CSV primary key value: {primary_key_value!r}")  # noqa: TRY003
        seen_keys.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


class CSVRowSourceAdapter:
    """Resolve row inputs into captured Sources and read them back."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one row as a JSON record with its exact citation identity."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSVRowSourceAdapter(),
    version=CSV_ROW_DEFINITION_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
