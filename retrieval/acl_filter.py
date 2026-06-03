"""Principal-based ACL filter applied AFTER RRF fusion, BEFORE the LLM.

The order matters:
  - Filtering before BM25 corrupts IDF statistics.
  - Filtering after the LLM is a data leak.
  - Filter post-retrieval, pre-generation, on the fused candidate set.
"""
from __future__ import annotations

from typing import Iterable

from .types import RetrievedChunk


def principals_for_user(user_id: str, groups: dict[str, list[str]] | None = None) -> set[str]:
    """Build the set of principals a user can match against chunk ACLs.

    A user's principals = {user:<id>} ∪ {group memberships} ∪ {team memberships}
                          ∪ {channel memberships} ∪ {"group:all-employees"}.
    """
    groups = groups or {}
    out: set[str] = {f"user:{user_id}", "group:all-employees", "workspace:default"}
    for principal, members in groups.items():
        if user_id in members:
            out.add(principal)
    return out


def filter_by_principals(
    candidates: Iterable[RetrievedChunk],
    user_principals: set[str],
    k_final: int,
) -> list[RetrievedChunk]:
    """Drop any chunk whose ACL set doesn't intersect the user's principal set."""
    out: list[RetrievedChunk] = []
    for c in candidates:
        chunk_acl = set(c.chunk.get("acl_principals") or [])
        if not chunk_acl:
            # Unrestricted chunk → allow (treat empty as public). Make this stricter in prod.
            out.append(c)
        elif chunk_acl & user_principals:
            out.append(c)
        if len(out) >= k_final:
            break
    return out
