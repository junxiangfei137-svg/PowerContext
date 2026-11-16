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

"""Chunked Context Packs and prefix deltas for incremental transport.

A full pack carries the original UTF-8 content as ordered chunks together with a digest
over the canonical unsigned JSON. A delta names the base digest, the number of base chunks
that can be reused as a prefix, and the target suffix. The receiver rebuilds the target pack
from the serialized delta alone; nothing is shared through process memory.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """A Context Pack or delta failed protocol or chunk validation."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete context split into ordered byte chunks."""

    items: tuple[bytes, ...]
    digest: str

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        """Build a pack from already chunked bytes."""

        normalized = tuple(items)
        for item in normalized:
            if not isinstance(item, bytes):
                raise ContextPackError("pack items must be bytes")  # noqa: TRY003
        return cls(items=normalized, digest=_digest_for(normalized))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split text into deterministic chunks that are valid UTF-8 and bounded in bytes."""

        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
        if not isinstance(content, str):
            raise ContextPackError("content must be text")  # noqa: TRY003
        if not content:
            return cls.build(())
        chunks: list[bytes] = []
        current = bytearray()
        for char in content:
            encoded = char.encode("utf-8")
            if len(encoded) > chunk_bytes:
                raise ContextPackError("a character does not fit within chunk_bytes")  # noqa: TRY003
            if len(current) + len(encoded) > chunk_bytes:
                chunks.append(bytes(current))
                current = bytearray()
            current.extend(encoded)
        if current:
            chunks.append(bytes(current))
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        payload = {**_unsigned_payload(self.items), "digest": self.digest}
        size = 0
        while True:
            data = _serialize({**payload, "byte_count": size})
            if len(data) == size:
                return data
            size = len(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        payload = _load_object(data, fields=_PACK_FIELDS, label="pack")
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("unsupported pack schema")  # noqa: TRY003
        items = _decode_items(payload["items"])
        digest = _validate_digest(payload["digest"])
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count != len(data):
            raise ContextPackError("pack byte_count does not match its serialized size")  # noqa: TRY003
        if digest != _digest_for(items):
            raise ContextPackError("pack digest does not match its items")  # noqa: TRY003
        return cls(items=items, digest=digest)

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def content_bytes(self) -> bytes:
        """Return the original content reassembled from the chunks."""

        return b"".join(self.items)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A reusable base prefix plus the target suffix that the receiver must append."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Reuse the longest common chunk prefix and carry the remaining target chunks."""

        prefix_items = 0
        shared = min(len(base.items), len(target.items))
        while prefix_items < shared and base.items[prefix_items] == target.items[prefix_items]:
            prefix_items += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix_items,
            appended=target.items[prefix_items:],
        )

    def to_bytes(self) -> bytes:
        return _serialize(
            {
                "schema": CONTEXT_PACK_DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": [base64.b64encode(item).decode("ascii") for item in self.appended],
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        payload = _load_object(data, fields=_DELTA_FIELDS, label="delta")
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("unsupported delta schema")  # noqa: TRY003
        base_digest = _validate_digest(payload["base_digest"])
        target_digest = _validate_digest(payload["target_digest"])
        prefix_items = payload["prefix_items"]
        if type(prefix_items) is not int or prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
        appended = _decode_items(payload["appended"])
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=appended,
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild and verify the target pack from a base pack and one delta."""

    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the supplied base pack")  # noqa: TRY003
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base chunk count")  # noqa: TRY003
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("rebuilt target digest does not match the delta")  # noqa: TRY003
    return target


def _unsigned_payload(items: tuple[bytes, ...]) -> dict[str, Any]:
    return {
        "schema": CONTEXT_PACK_SCHEMA,
        "items": [base64.b64encode(item).decode("ascii") for item in items],
    }


def _digest_for(items: tuple[bytes, ...]) -> str:
    canonical = _serialize(_unsigned_payload(items))
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _serialize(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_object(data: bytes, *, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise ContextPackError(f"{label} must be serialized bytes")  # noqa: TRY003
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError(f"{label} is not valid UTF-8 JSON") from error  # noqa: TRY003
    if not isinstance(payload, dict):
        raise ContextPackError(f"{label} must be a JSON object")  # noqa: TRY003
    keys = frozenset(payload)
    if keys != fields:
        raise ContextPackError(f"{label} fields do not match the schema")  # noqa: TRY003
    return payload


def _decode_items(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("chunk items must be a JSON array")  # noqa: TRY003
    decoded: list[bytes] = []
    for item in value:
        if not isinstance(item, str):
            raise ContextPackError("chunk items must be Base64 strings")  # noqa: TRY003
        try:
            decoded.append(base64.b64decode(item, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError("chunk item is not valid Base64") from error  # noqa: TRY003
    return tuple(decoded)


def _validate_digest(value: object) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ContextPackError("digest must be sha256:<64 hex characters>")  # noqa: TRY003
    return value


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
