"""Time-decay re-rank applied AFTER ACL filtering.

Each source has a configurable half-life:
  - Slack threads decay fast (most are stale after a month)
  - Notion pages decay slowly (process docs stay relevant)
  - Drive docs decay slowest (formal artifacts)

Score = fusion_score * 0.5 ** (age_days / half_life_days[source]).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable

from .types import RetrievedChunk


def _age_days(ts_iso: str, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    ts = datetime.fromisoformat(ts_iso)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (now - ts).total_seconds() / 86400.0)


def recency_rerank(
    chunks: Iterable[RetrievedChunk],
    half_life_days: dict[str, float],
    now: datetime | None = None,
) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    for c in chunks:
        source = c.chunk["source"]
        hl = half_life_days.get(source, 90.0)
        age = _age_days(c.chunk["timestamp_iso"], now=now)
        decay = math.pow(0.5, age / hl)
        out.append(RetrievedChunk(chunk_id=c.chunk_id, score=c.score * decay, chunk=c.chunk))
    out.sort(key=lambda c: c.score, reverse=True)
    return out
