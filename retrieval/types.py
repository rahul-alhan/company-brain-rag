"""Shared retrieval types — no heavy deps (faiss, openai) so ACL / recency
modules can be imported and tested on a box that only has stdlib + pandas."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    chunk: dict      # raw chunk record from chunks.jsonl
