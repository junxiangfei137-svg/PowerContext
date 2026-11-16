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

"""Capture one selected Pi session branch as immutable per-event Sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
    TextEvidence,
)
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_CONNECTOR_NAME = "pi-session-history"
PI_DEFINITION_VERSION = "1"
_CHECKPOINT_FIELD = "events"


class PiSessionInput(BaseModel):
    """Raw Pi session JSONL plus the optional selected leaf."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured immutable event evidence."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    source_id = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"pi event source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return source_id


def _decode_jsonl(content: bytes) -> list[dict[str, JsonValue]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be valid UTF-8") from error  # noqa: TRY003
    records: list[dict[str, JsonValue]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid Pi session JSON on line {line_number}") from error  # noqa: TRY003
        if not isinstance(record, dict):
            raise ValueError(f"Pi session line {line_number} must be a JSON object")  # noqa: TRY003
        records.append(record)
    return records


def _compact_json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text_parts(blocks: object) -> list[str]:
    parts: list[str] = []
    if isinstance(blocks, str):
        return [blocks]
    if not isinstance(blocks, list):
        return parts
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments", {})
            parts.append(f"[tool_call {name} {call_id}] {_compact_json(arguments)}")
    return parts


def _tool_result_text(blocks: object) -> list[str]:
    if isinstance(blocks, str):
        return [blocks]
    if not isinstance(blocks, list):
        return []
    return [
        block["text"]
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    ]


def _event_text(record: dict[str, JsonValue]) -> tuple[str, str, str] | None:
    """Return ``(event_type, role, content)`` for one visible record."""

    event_type = record.get("type")
    if event_type == "compaction" or event_type == "branch_summary":
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return str(event_type), str(event_type), f"{event_type}: {summary}"

    if event_type == "custom_message":
        parts = _text_parts(record.get("content"))
        if not parts:
            return None
        return event_type, "custom", "custom: " + "\n".join(parts)

    if event_type != "message":
        return None

    message = record.get("message")
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None

    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        body = "$ " + (command if isinstance(command, str) else "")
        if isinstance(output, str):
            body += "\n" + output
        return event_type, role, f"{role}: {body}"

    if role == "toolResult":
        tool_name = message.get("toolName")
        call_id = message.get("toolCallId")
        text = "\n".join(_tool_result_text(message.get("content")))
        return event_type, role, f"{role}: [tool_result {tool_name} {call_id}] {text}"

    parts = _text_parts(message.get("content"))
    if not parts:
        return None
    return event_type, role, f"{role}: " + "\n".join(parts)


def _parse_records(content: bytes) -> tuple[str, list[dict[str, JsonValue]]]:
    records = _decode_jsonl(content)
    if not records:
        raise ValueError("Pi session content must contain a session header")  # noqa: TRY003
    header, *events = records
    if header.get("type") != "session":
        raise ValueError("Pi session header must have type=session")  # noqa: TRY003
    if header.get("version") != 3:
        raise ValueError("Pi session header must have version=3")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Pi session header must have a non-empty id")  # noqa: TRY003

    by_id: dict[str, dict[str, JsonValue]] = {}
    order: list[str] = []
    for record in events:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("Pi session records must have non-empty ids")  # noqa: TRY003
        if event_id in by_id:
            raise ValueError(f"duplicate Pi session record id: {event_id!r}")  # noqa: TRY003
        if "parentId" not in record:
            raise ValueError(f"Pi session record {event_id!r} is missing parentId")  # noqa: TRY003
        parent_id = record.get("parentId")
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in by_id):
            raise ValueError(f"Pi session record {event_id!r} has an unknown parent")  # noqa: TRY003
        by_id[event_id] = record
        order.append(event_id)
    return session_id, [by_id[event_id] for event_id in order]


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse the selected branch and return its visible events root-first."""

    session_id, records = _parse_records(value.content)
    if not records:
        raise ValueError("Pi session has no records")  # noqa: TRY003
    if value.leaf_id is None:
        leaf_id = records[-1].get("id")
    else:
        leaf_id = value.leaf_id
    by_id = {record["id"]: record for record in records}
    if not isinstance(leaf_id, str) or leaf_id not in by_id:
        raise ValueError(f"Pi session leaf not found: {leaf_id!r}")  # noqa: TRY003

    chain: list[dict[str, JsonValue]] = []
    current: str | None = leaf_id
    seen: set[str] = set()
    while current is not None:
        if current in seen or current not in by_id:
            raise ValueError("Pi session branch is cyclic or incomplete")  # noqa: TRY003
        seen.add(current)
        record = by_id[current]
        chain.append(record)
        parent = record.get("parentId")
        current = parent if isinstance(parent, str) else None
    chain.reverse()

    events: list[PiEventInput] = []
    for record in chain:
        parsed = _event_text(record)
        if parsed is None:
            continue
        event_type, role, content = parsed
        event_id = record["id"]
        parent_id = record.get("parentId")
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=str(event_id),
                parent_id=parent_id if isinstance(parent_id, str) else None,
                event_type=event_type,
                role=role,
                content=content,
                record=record,
            )
        )
    return tuple(events)


class PiEventSourceAdapter:
    """Resolve and read immutable Pi event Sources."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
            materialization=SourceMaterialization.CAPTURED,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    version=PI_DEFINITION_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


def _record_digest(record: dict[str, JsonValue]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _checkpoint_events(checkpoint: JsonValue | None) -> dict[str, str]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("Pi connector checkpoint must be an object")  # noqa: TRY003
    events = checkpoint.get(_CHECKPOINT_FIELD, {})
    if not isinstance(events, dict):
        raise ValueError("Pi connector checkpoint events must be an object")  # noqa: TRY003
    result: dict[str, str] = {}
    for event_id, digest in events.items():
        if not isinstance(event_id, str) or not isinstance(digest, str):
            raise ValueError("Pi connector checkpoint entries must be strings")  # noqa: TRY003
        result[event_id] = digest
    return result


class PiSessionConnector:
    """Import only unseen events from one selected Pi session branch."""

    name = PI_CONNECTOR_NAME
    version = PI_DEFINITION_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: Path | str, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        events = parse_pi_session(PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id))
        imported = _checkpoint_events(session.checkpoint)

        for event in events:
            previous_digest = imported.get(event.event_id)
            current_digest = _record_digest(event.record)
            if previous_digest is not None and previous_digest != current_digest:
                raise ValueError(f"imported Pi event changed: {event.event_id!r}")  # noqa: TRY003

        for event in events:
            if event.event_id in imported:
                continue
            outcome = await session.submit(
                pi_event_source_id(event.session_id, event.event_id),
                PI_EVENT_SOURCE_NAME,
                event,
            )
            if outcome.status is not ConnectorSubmissionStatus.ACCEPTED:
                return ConnectorRunCompletion(status=ConnectorRunStatus.INCOMPLETE, checkpoint=session.checkpoint)
            imported[event.event_id] = _record_digest(event.record)

        checkpoint: JsonValue = {_CHECKPOINT_FIELD: dict(sorted(imported.items()))}
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


__all__ = [
    "PI_CONNECTOR_NAME",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
