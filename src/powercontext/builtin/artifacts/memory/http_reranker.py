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

"""Batched HTTP cross-encoder reranking for coarse Memory candidates."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ScoredRank(_StrictModel):
    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)


class _ScoreResponse(_StrictModel):
    scores: tuple[_ScoredRank, ...]


class HTTPMemoryReranker:
    """Score candidates with an external HTTP service and select by score.

    Candidates keep their original one-based rank across batches, so scores returned by
    separate requests are merged into one global ordering: descending score, then ascending
    rank.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")  # noqa: TRY003
        self._endpoint = str(endpoint)
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Send every candidate exactly once, then select global top ``limit`` ranks."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = tuple(enumerate(candidates[start : start + self._batch_size], start=start + 1))
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text} for rank, candidate in batch
                    ],
                },
            )
            response.raise_for_status()
            scores.update(_parse_scores(response, expected_ranks={rank for rank, _ in batch}))
            requests += 1
        ordered = sorted(scores, key=lambda rank: (-scores[rank], rank))
        return MemoryRerankDecision(
            selected_ranks=tuple(ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _parse_scores(response: httpx.Response, *, expected_ranks: set[int]) -> dict[int, float]:
    try:
        payload = _ScoreResponse.model_validate(response.json())
    except (ValueError, ValidationError) as error:
        raise ValueError("reranker response does not match the score schema") from error
    scores: dict[int, float] = {}
    for item in payload.scores:
        if item.rank not in expected_ranks or item.rank in scores:
            raise ValueError("reranker response ranks do not cover the request batch")  # noqa: TRY003
        scores[item.rank] = item.score
    if scores.keys() != expected_ranks:
        raise ValueError("reranker response ranks do not cover the request batch")  # noqa: TRY003
    return scores


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003


__all__ = ["DEFAULT_HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
