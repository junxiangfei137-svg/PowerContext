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

"""Durable dependency graph and last-event log for Source lifecycle propagation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_EVENTS_TABLE,
    SOURCE_LIFECYCLE_NODES_TABLE,
)


@dataclass(frozen=True, slots=True)
class SourceLifecycleNode:
    """One registered node of a Scope's evidence dependency graph."""

    node_id: str
    family: str
    policy: str | None
    parents: tuple[str, ...]
    revision: int
    available: bool
    reference: bytes

    @property
    def binding(self) -> tuple[str, str | None, tuple[str, ...], int, bytes]:
        """Everything registration pins; the propagated state is not part of it."""

        return (self.family, self.policy, self.parents, self.revision, self.reference)

    def with_availability(self, available: bool) -> SourceLifecycleNode:
        return SourceLifecycleNode(
            node_id=self.node_id,
            family=self.family,
            policy=self.policy,
            parents=self.parents,
            revision=self.revision,
            available=available,
            reference=self.reference,
        )


@dataclass(frozen=True, slots=True)
class SourceLifecycleEventRecord:
    sequence: int
    available: bool
    reason: str


class SourceLifecycleRepository:
    """Read and write the lifecycle graph inside one caller transaction."""

    async def load_nodes(self, connection: AsyncConnection, scope_id: str, /) -> dict[str, SourceLifecycleNode]:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        rows = (
            await connection.execute(
                select(
                    table.c.node_id,
                    table.c.family,
                    table.c.policy,
                    table.c.parents,
                    table.c.revision,
                    table.c.available,
                    table.c.reference,
                ).where(table.c.scope_id == scope_id)
            )
        ).all()
        return {
            row.node_id: SourceLifecycleNode(
                node_id=row.node_id,
                family=row.family,
                policy=row.policy,
                parents=tuple(json.loads(bytes(row.parents))),
                revision=int(row.revision),
                available=bool(row.available),
                reference=bytes(row.reference),
            )
            for row in rows
        }

    async def available_topic_artifacts(
        self,
        connection: AsyncConnection,
        scope_id: str,
        /,
    ) -> frozenset[str] | None:
        """Artifact ids the graph still supports, for read-time Topic eligibility.

        ``None`` means this Scope never registered Topic evidence, so no
        lifecycle filter applies.
        """

        table = SOURCE_LIFECYCLE_NODES_TABLE
        rows = (
            await connection.execute(
                select(table.c.node_id, table.c.available, table.c.reference).where(
                    table.c.scope_id == scope_id,
                    table.c.family == "topic",
                )
            )
        ).all()
        if not rows:
            return None
        available: set[str] = set()
        withheld: set[str] = set()
        for row in rows:
            reference = json.loads(bytes(row.reference))
            artifact_id = reference["artifact_id"]
            if bool(row.available):
                available.add(artifact_id)
            else:
                withheld.add(artifact_id)
        return frozenset(available - withheld)

    async def insert_node(self, connection: AsyncConnection, scope_id: str, node: SourceLifecycleNode, /) -> None:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        await connection.execute(
            insert(table).values(
                scope_id=scope_id,
                node_id=node.node_id,
                family=node.family,
                policy=node.policy,
                parents=json.dumps(node.parents).encode(),
                revision=node.revision,
                available=node.available,
                reference=node.reference,
            )
        )

    async def update_availability(
        self,
        connection: AsyncConnection,
        scope_id: str,
        states: Mapping[str, bool],
        /,
    ) -> None:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        for node_id in sorted(states):
            await connection.execute(
                update(table)
                .where(table.c.scope_id == scope_id, table.c.node_id == node_id)
                .values(available=states[node_id])
            )

    async def load_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        /,
    ) -> SourceLifecycleEventRecord | None:
        table = SOURCE_LIFECYCLE_EVENTS_TABLE
        row = (
            await connection.execute(
                select(table.c.sequence, table.c.available, table.c.reason).where(
                    table.c.scope_id == scope_id,
                    table.c.node_id == node_id,
                )
            )
        ).one_or_none()
        if row is None:
            return None
        return SourceLifecycleEventRecord(sequence=int(row.sequence), available=bool(row.available), reason=row.reason)

    async def save_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        record: SourceLifecycleEventRecord,
        /,
    ) -> None:
        table = SOURCE_LIFECYCLE_EVENTS_TABLE
        existing = await self.load_event(connection, scope_id, node_id)
        values = {
            "sequence": record.sequence,
            "available": record.available,
            "reason": record.reason,
        }
        if existing is None:
            await connection.execute(insert(table).values(scope_id=scope_id, node_id=node_id, **values))
            return
        await connection.execute(
            update(table)
            .where(table.c.scope_id == scope_id, table.c.node_id == node_id)
            .values(**values)
        )


__all__ = ["SourceLifecycleEventRecord", "SourceLifecycleNode", "SourceLifecycleRepository"]
