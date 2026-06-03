"""Retrieval package.

`RetrievedChunk`, `filter_by_principals`, `principals_for_user`, and
`recency_rerank` are import-cheap (stdlib + pandas only).

`HybridRetriever` is import-heavy (pulls faiss + openai). It is exposed via
the package namespace through a lazy attribute so `import retrieval` stays
fast and `retrieval.HybridRetriever` still works.
"""
from __future__ import annotations

from .types import RetrievedChunk
from .acl_filter import filter_by_principals, principals_for_user
from .recency import recency_rerank

__all__ = [
    "RetrievedChunk",
    "filter_by_principals",
    "principals_for_user",
    "recency_rerank",
    "HybridRetriever",
]


def __getattr__(name: str):
    if name == "HybridRetriever":
        from .hybrid import HybridRetriever
        return HybridRetriever
    raise AttributeError(f"module 'retrieval' has no attribute {name!r}")
