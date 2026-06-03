"""End-to-end ACL-aware RAG: query → hybrid retrieve → ACL filter → recency → LLM."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from anthropic import Anthropic

from retrieval import HybridRetriever, filter_by_principals, principals_for_user, recency_rerank
from .prompts import SYSTEM, USER_TEMPLATE, format_context


# Simple in-repo group registry. In production this comes from Okta / Workday / SCIM.
DEFAULT_GROUPS: dict[str, list[str]] = {
    "team:engineering": ["alice@acme.com", "bob@acme.com"],
    "team:security": ["alice@acme.com"],
    "channel:eng-auth-migration": ["alice@acme.com", "bob@acme.com"],
    "channel:eng-auth-migration:private": ["alice@acme.com", "bob@acme.com"],
    "team:design": ["carol@acme.com"],
    "team:contractors": ["contractor@acme.com"],
}


@dataclass
class BrainAnswer:
    answer: str
    citations: list[str]
    user: str


def answer_for(user: str, query: str, cfg: dict, groups: dict | None = None) -> BrainAnswer:
    groups = groups or DEFAULT_GROUPS
    retriever = HybridRetriever(cfg["index_dir"], embed_model=cfg["embedding"]["model"])

    raw = retriever.search(
        query=query,
        k_dense=cfg["retrieval"]["k_dense"],
        k_sparse=cfg["retrieval"]["k_sparse"],
        k_final=cfg["retrieval"]["k_final"],
        rrf_k=cfg["retrieval"]["rrf_k"],
    )

    user_princ = principals_for_user(user, groups)
    filtered = filter_by_principals(raw, user_princ, k_final=cfg["retrieval"]["k_final"] * 2)

    if cfg["recency"]["enabled"]:
        filtered = recency_rerank(filtered, cfg["recency"]["half_life_days"])
    final = filtered[: cfg["retrieval"]["k_final"]]

    if not final:
        return BrainAnswer(
            answer="I couldn't find any documents you have access to that answer this. (No retrievable context.)",
            citations=[],
            user=user,
        )

    ctx = format_context(final)
    client = Anthropic()
    msg = client.messages.create(
        model=cfg["generation"]["model"],
        max_tokens=cfg["generation"]["max_tokens"],
        system=SYSTEM,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(context=ctx, query=query)}],
    )
    text = msg.content[0].text
    citations = [c.chunk_id for c in final]
    return BrainAnswer(answer=text, citations=citations, user=user)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    a = answer_for(args.user, args.query, cfg)
    print(json.dumps({"user": a.user, "answer": a.answer, "citations": a.citations}, indent=2))


if __name__ == "__main__":
    main()
