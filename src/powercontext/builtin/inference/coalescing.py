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

"""Share embedding work across concurrent callers and reuse finished vectors."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


class _Caller:
    """One ``embed`` call: its result slots, its pending texts, and its usage."""

    __slots__ = ("error", "future", "input_tokens", "output_tokens", "pending", "requests", "vectors")

    def __init__(self) -> None:
        self.future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.pending = 0
        self.vectors: dict[str, EmbeddingVector] = {}
        self.requests = 0
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.error: BaseException | None = None


class _Flight:
    """One distinct text that a batch will compute exactly once."""

    __slots__ = ("callers", "text")

    def __init__(self, text: str) -> None:
        self.text = text
        self.callers: list[_Caller] = []


class CoalescingEmbeddingModel:
    """Deduplicate concurrent embedding work and cache successful vectors.

    Texts requested while a provider request is still running join that
    request instead of issuing another one, so a caller only ever pays for the
    distinct batches it actually waited on. A cancelled caller detaches from
    its own future only; the shared provider work belongs to the wrapper.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)):
            raise ValueError("coalesce_ms must be a non-negative number")  # noqa: TRY003
        if not math.isfinite(coalesce_ms) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")  # noqa: TRY003
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._flights: dict[str, _Flight] = {}
        self._pending: OrderedDict[str, _Flight] = OrderedDict()
        self._flush_task: asyncio.Task[None] | None = None

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        unique = tuple(dict.fromkeys(texts))
        if not unique:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        vectors: dict[str, EmbeddingVector] = {}
        missing: list[str] = []
        for text in unique:
            cached = self._cached(text)
            if cached is None:
                missing.append(text)
            else:
                vectors[text] = cached
        if not missing:
            return EmbeddingResult(
                vectors=tuple(vectors[text] for text in texts),
                usage=InferenceUsage(requests=0),
            )
        caller = _Caller()
        for text in missing:
            flight = self._flights.get(text)
            if flight is None:
                flight = _Flight(text)
                self._flights[text] = flight
                self._pending[text] = flight
            flight.callers.append(caller)
            caller.pending += 1
        self._start_flush()
        await asyncio.shield(caller.future)
        if caller.error is not None:
            raise caller.error
        vectors.update(caller.vectors)
        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in texts),
            usage=InferenceUsage(
                requests=caller.requests,
                input_tokens=caller.input_tokens,
                output_tokens=caller.output_tokens,
            ),
        )

    def _cached(self, text: str) -> EmbeddingVector | None:
        if self._cache_size <= 0:
            return None
        vector = self._cache.get(text)
        if vector is not None:
            self._cache.move_to_end(text)
        return vector

    def _start_flush(self) -> None:
        loop = asyncio.get_running_loop()
        task = self._flush_task
        if task is not None and not task.done() and task.get_loop() is loop:
            return
        self._flush_task = loop.create_task(self._flush())

    async def _flush(self) -> None:
        try:
            await asyncio.sleep(self._coalesce_seconds)
            while self._pending:
                batch = self._claim()
                if not batch:
                    break
                await self._embed_batch(batch)
        finally:
            self._flush_task = None

    def _claim(self) -> tuple[_Flight, ...]:
        batch: list[_Flight] = []
        while self._pending and len(batch) < self._max_batch_size:
            _, flight = self._pending.popitem(last=False)
            batch.append(flight)
        return tuple(batch)

    async def _embed_batch(self, flights: tuple[_Flight, ...]) -> None:
        texts = tuple(flight.text for flight in flights)
        try:
            result = await self._model.embed(texts)
        except asyncio.CancelledError:
            self._discard(flights, cancelled=True)
            raise
        except Exception as error:  # noqa: BLE001 - one provider failure must stay local to its batch
            self._discard(flights, error=error)
            return
        if len(result.vectors) != len(flights):
            self._discard(flights, error=ValueError("embedding provider returned a mismatched vector count"))
            return
        for flight, vector in zip(flights, result.vectors, strict=True):
            self._flights.pop(flight.text, None)
            self._remember(flight.text, vector)
            for caller in flight.callers:
                caller.vectors[flight.text] = vector
        involved: dict[int, _Caller] = {}
        for flight in flights:
            for caller in flight.callers:
                involved.setdefault(id(caller), caller)
        for caller in involved.values():
            caller.requests += 1
            if result.usage.input_tokens is not None:
                caller.input_tokens = (caller.input_tokens or 0) + result.usage.input_tokens
            if result.usage.output_tokens is not None:
                caller.output_tokens = (caller.output_tokens or 0) + result.usage.output_tokens
        for flight in flights:
            self._settle(flight, error=None, cancelled=False)

    def _remember(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size <= 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _discard(
        self,
        flights: tuple[_Flight, ...],
        *,
        error: BaseException | None = None,
        cancelled: bool = False,
    ) -> None:
        for flight in flights:
            self._flights.pop(flight.text, None)
            self._settle(flight, error=error, cancelled=cancelled)

    @staticmethod
    def _settle(flight: _Flight, *, error: BaseException | None, cancelled: bool) -> None:
        for caller in flight.callers:
            if error is not None and caller.error is None:
                caller.error = error
            caller.pending -= 1
            if caller.pending > 0 or caller.future.done():
                continue
            if cancelled:
                caller.future.cancel()
            else:
                caller.future.set_result(None)


__all__ = ["CoalescingEmbeddingModel"]
