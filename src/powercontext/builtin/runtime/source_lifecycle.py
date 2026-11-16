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

"""Propagate Source withdrawal and restoration through registered evidence.

A Scope registers an explicit evidence dependency graph: Sources at the
roots, Memory and Topic Memory nodes below them. Lifecycle events only ever
target a Source; every derived node is re-evaluated from its ``all`` / ``any``
policy. Deactivation changes eligibility alone, so bodies, revision history,
and exact audit references stay intact.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.source_lifecycle import (
    SourceLifecycleEventRecord,
    SourceLifecycleNode,
    SourceLifecycleRepository,
)
from powercontext.builtin.persistence.tables import MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.builtin.sources import validate_scope_id
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

LIFECYCLE_PURPOSE_GENERATION = "generation"
LIFECYCLE_PURPOSE_AUDIT = "audit"
_LIFECYCLE_PURPOSES = frozenset({LIFECYCLE_PURPOSE_GENERATION, LIFECYCLE_PURPOSE_AUDIT})
_SOURCE_FAMILY = "source"
_MEMORY_FAMILY = "memory"
_TOPIC_FAMILY = "topic"
_POLICY_ALL = "all"
_POLICY_ANY = "any"
_POLICIES = frozenset({_POLICY_ALL, _POLICY_ANY})
_REFERENCE_KIND = {_SOURCE_FAMILY: "source", _MEMORY_FAMILY: "memory", _TOPIC_FAMILY: "topic"}
_DEACTIVATION_REASON = "source-lifecycle"


class SourceLifecycleError(ValueError):
    """Reject an invalid registration, event, or resolution request."""


class SourceLifecycleEvent(BaseModel):
    """One validated lifecycle transition for a single registered Source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: StrictInt = Field(ge=1)
    sequence: StrictInt = Field(ge=1)
    available: StrictBool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")  # noqa: TRY003
        return value.strip()


class ScopedSourceLifecycleApplication:
    """Register one Scope's evidence graph and apply Source lifecycle events."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = validate_scope_id(scope_id)
        self._repository = SourceLifecycleRepository()

    async def register_source(
        self,
        node_id: str,
        reference: SourceRef,
        /,
        *,
        revision: int = 1,
    ) -> None:
        node = _validate_node_id(node_id)
        if not isinstance(reference, SourceRef):
            raise SourceLifecycleError("a Source node requires a SourceRef")  # noqa: TRY003
        if type(revision) is not int or revision < 1:
            raise SourceLifecycleError("revision must be a positive integer")  # noqa: TRY003
        await self._runtime.records.for_scope(self.scope_id).get_source(
            reference.source_type,
            reference.source_id,
        )
        record = SourceLifecycleNode(
            node_id=node,
            family=_SOURCE_FAMILY,
            policy=None,
            parents=(),
            revision=revision,
            available=True,
            reference=_encode_reference(_SOURCE_FAMILY, reference),
        )
        async with self._runtime._context(self.scope_id):
            async with self._runtime._locked(self.scope_id):
                async with self._runtime.database.transaction() as connection:
                    existing = (await self._repository.load_nodes(connection, self.scope_id)).get(node)
                    if existing is not None:
                        if existing.binding != record.binding:
                            raise SourceLifecycleError("node id is already registered with another binding")  # noqa: TRY003
                        return
                    await self._repository.insert_node(connection, self.scope_id, record)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: Iterable[str],
        policy: str = _POLICY_ALL,
    ) -> None:
        node = _validate_node_id(node_id)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("a Memory node requires a MemoryCitation")  # noqa: TRY003
        dependency = _validate_dependencies(node, parents, policy)
        await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=citation))
        record = SourceLifecycleNode(
            node_id=node,
            family=_MEMORY_FAMILY,
            policy=dependency.policy,
            parents=dependency.parents,
            revision=0,
            available=True,
            reference=_encode_reference(_MEMORY_FAMILY, citation),
        )
        await self._register_derived(record)

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        /,
        *,
        parents: Iterable[str],
        policy: str = _POLICY_ALL,
    ) -> None:
        node = _validate_node_id(node_id)
        if not isinstance(reference, ArtifactRef):
            raise SourceLifecycleError("a Topic node requires an ArtifactRef")  # noqa: TRY003
        if reference.family != "topic-memory":
            raise SourceLifecycleError("a Topic node requires a topic-memory reference")  # noqa: TRY003
        dependency = _validate_dependencies(node, parents, policy)
        await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
            reference.family,
            reference.artifact_id,
            reference.revision,
        )
        record = SourceLifecycleNode(
            node_id=node,
            family=_TOPIC_FAMILY,
            policy=dependency.policy,
            parents=dependency.parents,
            revision=0,
            available=True,
            reference=_encode_reference(_TOPIC_FAMILY, reference),
        )
        await self._register_derived(record)

    async def apply(self, event: SourceLifecycleEvent | Mapping[str, Any], /) -> list[str]:
        """Propagate one event, returning every node still available afterwards."""

        if not isinstance(event, SourceLifecycleEvent):
            event = SourceLifecycleEvent.model_validate(event)
        async with self._runtime._context(self.scope_id):
            async with self._runtime._locked(self.scope_id):
                async with self._runtime.database.transaction() as connection:
                    nodes = await self._repository.load_nodes(connection, self.scope_id)
                    previous = await self._repository.load_event(connection, self.scope_id, event.node_id)
                node = nodes.get(event.node_id)
                if node is None or node.family != _SOURCE_FAMILY:
                    raise SourceLifecycleError("event targets an unregistered Source")  # noqa: TRY003
                if event.revision != node.revision:
                    raise SourceLifecycleError("event revision does not match the registered Source revision")  # noqa: TRY003
                if previous is not None:
                    if event.sequence < previous.sequence:
                        raise SourceLifecycleError("event sequence must not move backwards")  # noqa: TRY003
                    if event.sequence == previous.sequence:
                        if event.available != previous.available or event.reason != previous.reason:
                            raise SourceLifecycleError("event sequence is already recorded with other content")  # noqa: TRY003
                        return sorted(node_id for node_id, item in nodes.items() if item.available)
                states = _propagate(nodes, event.node_id, event.available)
                pending = {
                    node_id: available for node_id, available in states.items() if nodes[node_id].available != available
                }
                # Eligibility is written before the graph state: a failure here leaves the
                # event unrecorded, so retrying the same event finishes the propagation.
                for node_id in sorted(pending):
                    if nodes[node_id].family == _MEMORY_FAMILY:
                        await self._set_memory_availability(nodes[node_id].reference, available=pending[node_id])
                async with self._runtime.database.transaction() as connection:
                    await self._repository.update_availability(connection, self.scope_id, pending)
                    await self._repository.save_event(
                        connection,
                        self.scope_id,
                        event.node_id,
                        SourceLifecycleEventRecord(
                            sequence=event.sequence,
                            available=event.available,
                            reason=event.reason,
                        ),
                    )
        return sorted(node_id for node_id, available in states.items() if available)

    async def resolve(
        self,
        node_id: str,
        /,
        *,
        purpose: str = LIFECYCLE_PURPOSE_GENERATION,
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        """Return the registered reference, respecting generation eligibility."""

        node = _validate_node_id(node_id)
        if purpose not in _LIFECYCLE_PURPOSES:
            raise SourceLifecycleError("unknown resolution purpose")  # noqa: TRY003
        async with self._runtime._context(self.scope_id):
            async with self._runtime.database.transaction() as connection:
                nodes = await self._repository.load_nodes(connection, self.scope_id)
        record = nodes.get(node)
        if record is None:
            raise SourceLifecycleError("unknown lifecycle node")  # noqa: TRY003
        if purpose == LIFECYCLE_PURPOSE_GENERATION and not record.available:
            raise SourceLifecycleError("node is not available for generation")  # noqa: TRY003
        return _decode_reference(record.reference)

    async def _register_derived(self, record: SourceLifecycleNode) -> None:
        async with self._runtime._context(self.scope_id):
            async with self._runtime._locked(self.scope_id):
                async with self._runtime.database.transaction() as connection:
                    nodes = await self._repository.load_nodes(connection, self.scope_id)
                    existing = nodes.get(record.node_id)
                    if existing is not None:
                        if existing.binding != record.binding:
                            raise SourceLifecycleError("node id is already registered with another binding")  # noqa: TRY003
                        return
                    missing = tuple(parent for parent in record.parents if parent not in nodes)
                    if missing:
                        raise SourceLifecycleError("parents must be registered before their dependants")  # noqa: TRY003
                    available = _evaluate(record.policy, tuple(nodes[parent].available for parent in record.parents))
                    stored = record.with_availability(available)
                    await self._repository.insert_node(connection, self.scope_id, stored)
            if not available and stored.family == _MEMORY_FAMILY:
                await self._set_memory_availability(stored.reference, available=False)

    async def _set_memory_availability(self, reference: bytes, *, available: bool) -> None:
        citation = _decode_reference(reference)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("a Memory node requires a MemoryCitation")  # noqa: TRY003
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            current = await service.head(context.artifacts.memory_artifact_id)
            item = next(
                (entry for entry in current.content.manifest.entries if entry.entry_id == citation.entry_id),
                None,
            )
            if item is None:
                raise SourceLifecycleError("registered Memory entry no longer exists")  # noqa: TRY003
            if item.entry_version_id != citation.entry_version_id:
                raise SourceLifecycleError("registered Memory entry body changed after registration")  # noqa: TRY003
            if (item.state == "active") == available:
                return
            entry = next(entry for entry in await service.entries(current) if entry.entry_id == citation.entry_id)
            if available:
                await service.reactivate(current, entries=(entry,), reason=_DEACTIVATION_REASON)
            else:
                await service.forget(current, entries=(entry,), reason=_DEACTIVATION_REASON)


class SourceLifecycleApplication:
    """Select the lifecycle service for one Scope."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleApplication:
        return ScopedSourceLifecycleApplication(self._runtime, scope_id)


class _Dependencies(BaseModel):
    parents: tuple[str, ...]
    policy: str


def _validate_node_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("node id must be a non-empty string")  # noqa: TRY003
    node = value.strip()
    if len(node) > MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH:
        raise SourceLifecycleError("node id is too long")  # noqa: TRY003
    return node


def _validate_dependencies(node_id: str, parents: Iterable[str], policy: str) -> _Dependencies:
    if isinstance(parents, str):
        raise SourceLifecycleError("parents must be a collection of node ids")  # noqa: TRY003
    items = tuple(_validate_node_id(parent) for parent in parents)
    if not items:
        raise SourceLifecycleError("a derived node requires at least one parent")  # noqa: TRY003
    if len(set(items)) != len(items):
        raise SourceLifecycleError("parents must be unique")  # noqa: TRY003
    if node_id in items:
        raise SourceLifecycleError("a node cannot depend on itself")  # noqa: TRY003
    if policy not in _POLICIES:
        raise SourceLifecycleError("policy must be 'all' or 'any'")  # noqa: TRY003
    return _Dependencies(parents=items, policy=policy)


def _evaluate(policy: str, support: tuple[bool, ...], /) -> bool:
    return all(support) if policy == _POLICY_ALL else any(support)


def _propagate(
    nodes: Mapping[str, SourceLifecycleNode],
    source_id: str,
    available: bool,
    /,
) -> dict[str, bool]:
    states = {node_id: node.available for node_id, node in nodes.items()}
    states[source_id] = available
    for node_id in _topological_order(nodes):
        node = nodes[node_id]
        if node.family == _SOURCE_FAMILY:
            continue
        states[node_id] = _evaluate(node.policy or _POLICY_ALL, tuple(states[parent] for parent in node.parents))
    return states


def _topological_order(nodes: Mapping[str, SourceLifecycleNode]) -> tuple[str, ...]:
    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    pending = {node_id: 0 for node_id in nodes}
    for node_id, node in nodes.items():
        for parent in node.parents:
            children.setdefault(parent, []).append(node_id)
            pending[node_id] += 1
    ready = deque(sorted(node_id for node_id, count in pending.items() if count == 0))
    order: list[str] = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for child in sorted(children.get(current, ())):
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        raise SourceLifecycleError("dependency graph contains a cycle")  # noqa: TRY003
    return tuple(order)


def _encode_reference(family: str, reference: SourceRef | MemoryCitation | ArtifactRef) -> bytes:
    payload = {"kind": _REFERENCE_KIND[family], **reference.model_dump(mode="json")}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _decode_reference(payload: bytes) -> SourceRef | MemoryCitation | ArtifactRef:
    value = json.loads(bytes(payload))
    kind = value.pop("kind", None)
    if kind == _SOURCE_FAMILY:
        return SourceRef.model_validate(value)
    if kind == _MEMORY_FAMILY:
        return MemoryCitation.model_validate(value)
    if kind == _TOPIC_FAMILY:
        return ArtifactRef.model_validate(value)
    raise SourceLifecycleError("stored reference is not a lifecycle reference")  # noqa: TRY003


__all__ = [
    "LIFECYCLE_PURPOSE_AUDIT",
    "LIFECYCLE_PURPOSE_GENERATION",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "ScopedSourceLifecycleApplication",
]
