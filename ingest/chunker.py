"""Recursive token-aware chunker.

Source-aware overrides:
  - Slack threads are short → chunk at thread boundary, no further split
  - Notion / Drive → recursive paragraph → sentence → word
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import tiktoken

from connectors.base import Document


@dataclass
class Chunk:
    chunk_id: str
    parent_doc_id: str
    source: str
    text: str
    timestamp_iso: str
    acl_principals: list[str]
    title: str
    position: int = 0
    extra: dict = field(default_factory=dict)


_ENC = tiktoken.get_encoding("cl100k_base")


def _tok_len(s: str) -> int:
    return len(_ENC.encode(s))


def _split_recursive(text: str, size: int, overlap: int) -> list[str]:
    seps = ["\n\n", "\n", ". ", " "]
    if _tok_len(text) <= size:
        return [text]

    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks: list[str] = []
        buf = ""
        for p in parts:
            candidate = (buf + sep + p) if buf else p
            if _tok_len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        if overlap and len(chunks) > 1:
            chunks = _apply_overlap(chunks, overlap)
        return chunks
    # fallback: token-level slice
    toks = _ENC.encode(text)
    return [_ENC.decode(toks[i:i + size]) for i in range(0, len(toks), max(1, size - overlap))]


def _apply_overlap(chunks: list[str], overlap_toks: int) -> list[str]:
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        prev_tail = _ENC.decode(_ENC.encode(prev)[-overlap_toks:])
        out.append(prev_tail + " " + cur)
    return out


def chunk_document(doc: Document, size: int, overlap: int) -> Iterable[Chunk]:
    if doc.source == "slack":
        # Threads are short; preserve as one chunk
        yield Chunk(
            chunk_id=f"{doc.doc_id}::0",
            parent_doc_id=doc.doc_id,
            source=doc.source,
            text=doc.text,
            timestamp_iso=doc.timestamp.isoformat(),
            acl_principals=list(doc.acl_principals),
            title=doc.title,
            position=0,
            extra=dict(doc.extra),
        )
        return

    for i, piece in enumerate(_split_recursive(doc.text, size, overlap)):
        yield Chunk(
            chunk_id=f"{doc.doc_id}::{i}",
            parent_doc_id=doc.doc_id,
            source=doc.source,
            text=piece,
            timestamp_iso=doc.timestamp.isoformat(),
            acl_principals=list(doc.acl_principals),
            title=doc.title,
            position=i,
            extra=dict(doc.extra),
        )
