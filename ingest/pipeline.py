"""End-to-end ingest: connect → normalize → chunk → embed → persist.

Persists three artifacts under index/:
  - chunks.jsonl        (all chunk metadata, source-of-truth)
  - dense.faiss         (FAISS IndexFlatIP over embeddings)
  - bm25.pkl            (rank_bm25 BM25Okapi over tokenized chunk text)
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import yaml
from openai import OpenAI
from rank_bm25 import BM25Okapi

from connectors import SlackConnector, NotionConnector, DriveConnector
from .chunker import Chunk, chunk_document


_WORD = re.compile(r"\w+")


def _tokenize(s: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(s)]


def _embed(client: OpenAI, model: str, texts: list[str]) -> np.ndarray:
    # OpenAI embeddings API accepts batches; chunk to be safe
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        r = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in r.data])
    arr = np.asarray(out, dtype=np.float32)
    # cosine via IP on L2-normalized vectors
    faiss.normalize_L2(arr)
    return arr


def run(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    index_dir = Path(cfg["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)

    connectors = [
        SlackConnector(path=cfg["sources"]["slack"]),
        NotionConnector(path=cfg["sources"]["notion"]),
        DriveConnector(path=cfg["sources"]["drive"]),
    ]

    chunks: list[Chunk] = []
    for c in connectors:
        for doc in c.fetch():
            for ch in chunk_document(
                doc,
                size=cfg["chunking"]["size_tokens"],
                overlap=cfg["chunking"]["overlap_tokens"],
            ):
                chunks.append(ch)
    print(f"Chunked {len(chunks)} chunks from {len(connectors)} sources")

    # Persist chunk metadata
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__) + "\n")

    # BM25 over tokenized text
    tokenized = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with (index_dir / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": [c.chunk_id for c in chunks]}, f)
    print(f"Built BM25 index over {len(chunks)} chunks")

    # Dense embeddings
    client = OpenAI()
    emb = _embed(client, cfg["embedding"]["model"], [c.text for c in chunks])
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(index_dir / "dense.faiss"))
    (index_dir / "dense.ids.json").write_text(json.dumps([c.chunk_id for c in chunks]))
    print(f"Built FAISS index: dim={emb.shape[1]}, n={emb.shape[0]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
