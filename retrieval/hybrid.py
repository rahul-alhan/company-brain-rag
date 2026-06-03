"""Hybrid retrieval: BM25 + dense, fused via Reciprocal Rank Fusion.

RRF is parameter-free across scoring scales — the right choice when BM25 raw
scores and cosine similarities aren't comparable.

`faiss` and `openai` are imported lazily inside the class so the module
(and the pure-Python `rrf_fuse` helper) can be imported and unit-tested
without those heavy deps installed.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np

from .types import RetrievedChunk


_WORD = re.compile(r"\w+")


def _tokenize(s: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(s)]


def rrf_fuse(*ranked_lists: list[str], rrf_k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Pass any number of ranked id-lists; returns
    [(id, fused_score), ...] sorted by score desc. Pure Python — no heavy deps."""
    fused: dict[str, float] = {}
    for ids in ranked_lists:
        for rank, cid in enumerate(ids):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, index_dir: str, embed_model: str = "text-embedding-3-small"):
        import faiss                                # lazy
        from openai import OpenAI                   # lazy

        idx = Path(index_dir)
        self.chunks: dict[str, dict] = {}
        for line in (idx / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            self.chunks[r["chunk_id"]] = r

        with (idx / "bm25.pkl").open("rb") as f:
            blob = pickle.load(f)
            self.bm25 = blob["bm25"]
            self.bm25_ids = blob["ids"]

        self.dense = faiss.read_index(str(idx / "dense.faiss"))
        self.dense_ids = json.loads((idx / "dense.ids.json").read_text())

        self.embed_model = embed_model
        self.client = OpenAI()

    def _embed_query(self, q: str) -> np.ndarray:
        import faiss                                # lazy
        r = self.client.embeddings.create(model=self.embed_model, input=[q])
        v = np.asarray([r.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(v)
        return v

    def search(self, query: str, k_dense: int, k_sparse: int, k_final: int,
               rrf_k: int = 60) -> list[RetrievedChunk]:
        # Sparse
        tokens = _tokenize(query)
        bm_scores = self.bm25.get_scores(tokens)
        sparse_top = np.argsort(bm_scores)[::-1][:k_sparse]
        sparse_ids = [self.bm25_ids[i] for i in sparse_top]

        # Dense — FAISS returns -1 in slots where the index couldn't fill k_dense
        # (e.g. k_dense > index size, or IVF with no nprobe hit). Indexing
        # self.dense_ids[-1] would silently inject the last chunk as a bogus hit.
        qv = self._embed_query(query)
        _, dense_top = self.dense.search(qv, k_dense)
        dense_ids = [self.dense_ids[i] for i in dense_top[0] if i != -1]

        ranked = rrf_fuse(sparse_ids, dense_ids, rrf_k=rrf_k)[:k_final * 4]
        return [RetrievedChunk(chunk_id=cid, score=s, chunk=self.chunks[cid]) for cid, s in ranked]
