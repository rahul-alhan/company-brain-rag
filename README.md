# Company Brain RAG — Multi-Source Enterprise Knowledge with ACL-Aware Retrieval

Production-style **enterprise RAG** over Slack + Notion + Drive (synthetic) with **per-user access control** enforced at retrieval time, **hybrid sparse+dense** fusion, and **recency decay** for time-sensitive corporate knowledge.

> The "Company Brain" pattern in YC's 2026 RFS — a single semantic layer over everything a company writes down. The hard part isn't retrieval, it's making sure Alice never retrieves a doc that only Bob has access to.

---

## Why ACL-First RAG

Most RAG demos index a corpus and answer questions over it. That's fine for a public help center; it's a **data leak** the moment two users with different access rights share an index.

This repo treats access control as a first-class retrieval concern, not a post-hoc filter on the LLM output:

1. Every ingested chunk carries an `acl_principals: list[str]` field (groups + user IDs that can see it).
2. The retriever **filters before re-ranking** — never wastes context on docs the requester can't see.
3. Citations carry the source ACL — auditors can prove no cross-tenant bleed.
4. A red-team eval set asserts that user A's queries never surface user B's private docs.

---

## Architecture

```
   Slack       Notion      Drive
     │           │           │
     ▼           ▼           ▼
  ┌────────────────────────────┐
  │   Connectors (ABC)         │   ──▶  Normalize → Document schema
  │   - principals[]           │
  │   - source / source_id     │
  │   - timestamp              │
  └────────────────────────────┘
                │
                ▼
        Chunker (recursive, 384 tok, 64 overlap)
                │
                ▼
   ┌──────────────────┐      ┌─────────────────┐
   │ BM25 (rank_bm25) │      │  Dense (FAISS)  │
   └──────────────────┘      └─────────────────┘
                │                    │
                └──────────┬─────────┘
                           ▼
              Reciprocal Rank Fusion (RRF)
                           │
                           ▼
                ACL filter (post-fusion)
                           │
                           ▼
                Recency decay re-rank
                           │
                           ▼
              ┌──────────────────────┐
              │  brain.pipeline      │  ──▶  Claude / GPT-4 + cited answer
              └──────────────────────┘
                           │
                           ▼
                   harness.py
              (faithfulness, cross-source
               synthesis, ACL leak rate)
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Ingest the synthetic multi-source corpus
python -m ingest.pipeline --config configs/default.yaml

# 2. Ask as a specific user (ACL applies)
python -m brain.pipeline \
  --user alice@acme.com \
  --query "What did engineering decide about the auth migration?"

# 3. Same query as a user without auth-team access — different answer
python -m brain.pipeline \
  --user contractor@acme.com \
  --query "What did engineering decide about the auth migration?"

# 4. Run the eval harness (quality + ACL leak rate)
python -m eval.harness --eval-set eval/eval_set.json
```

---

## What Makes This Different from a Generic RAG Demo

| Concern | Generic RAG | Company Brain |
|---|---|---|
| **Access control** | Whole-corpus access | Per-chunk ACL, filtered pre-LLM |
| **Source diversity** | Single doc type | Slack threads, Notion pages, Drive docs — all normalized |
| **Recency** | Treated equally | Time decay (`half_life_days` configurable per source) |
| **Citation** | Doc name | `(source: notion::page-id::block-id, last_edited: 2026-04-22)` |
| **Eval** | Faithfulness only | + cross-source synthesis + ACL leak rate (must be 0) |

---

## Evaluation Results (synthetic corpus, 240 docs, 8 users, 3 sources)

| Metric | Score |
|---|---|
| Faithfulness | 0.89 |
| Cross-source synthesis (2+ sources cited correctly) | 0.81 |
| Answer relevance | 0.86 |
| **ACL leak rate** (private docs surfacing to unauthorized users) | **0.0%** |
| Recency-weighted precision @ k=6 | 0.84 |

> Numbers above are illustrative; actual values depend on corpus and eval set. The ACL leak rate is a hard gate — any non-zero value blocks promotion.

---

## Repository Layout

```
company-brain-rag/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/
│   └── default.yaml
├── connectors/
│   ├── __init__.py
│   ├── base.py                # Connector ABC + normalized Document schema
│   ├── slack_connector.py     # threads → Documents (channel-based ACL)
│   ├── notion_connector.py    # pages/blocks → Documents (workspace+page ACL)
│   └── drive_connector.py     # docs → Documents (file-share ACL)
├── ingest/
│   ├── __init__.py
│   ├── chunker.py             # recursive splitter (source-aware)
│   └── pipeline.py            # connect → normalize → chunk → embed → persist
├── retrieval/
│   ├── __init__.py
│   ├── hybrid.py              # BM25 + dense fusion via RRF
│   ├── acl_filter.py          # principal-based filtering
│   └── recency.py             # time-decay re-rank
├── brain/
│   ├── __init__.py
│   ├── pipeline.py            # full RAG with citation enforcement
│   └── prompts.py             # system + answer templates
├── eval/
│   ├── harness.py             # faithfulness + cross-source + ACL leak
│   └── eval_set.json          # ground-truth Q/A with per-user expected answers
├── data/                      # tiny synthetic corpora (committed)
│   ├── slack_sample.jsonl
│   ├── notion_sample.jsonl
│   └── drive_sample.jsonl
└── tests/
    ├── test_acl_filter.py
    ├── test_hybrid.py
    └── test_recency.py
```

---

## Design Choices

| Decision | Rationale |
|---|---|
| **ACL filter post-RRF, pre-LLM** | Filtering pre-retrieval breaks BM25 IDF stats; filtering post-LLM is a leak. RRF then ACL is the correct order. |
| **RRF over weighted-sum fusion** | RRF is parameter-free and robust across scoring scales (BM25 raw vs cosine sim) |
| **FAISS, not a hosted vector DB** | Demo-portable; production swap = pgvector / OpenSearch / Pinecone |
| **Time decay as re-rank, not score boost** | Re-rank step is explicit and tunable; baked-in score boosts hide the recency contribution |
| **Synthetic corpus committed to repo** | Reproducible by anyone; no real company data risk |
| **Eval has a zero-tolerance gate (ACL leak)** | Quality metrics are trade-offs; leak rate is binary correctness |

---

## Production Notes

Real deployments of this pattern require:

- **Live ACL sync** — Slack/Notion/Drive share-graph changes hourly; stale ACLs are how leaks happen
- **pgvector or OpenSearch hybrid** — FAISS in this repo is single-node; production needs distributed
- **Embedding-time PII redaction** — separate pipeline; out of scope here
- **Audit log of every retrieval** — `(user, query, chunk_ids, ACL_decisions)` for compliance
- **Per-source rate limits** during ingest — Slack/Notion APIs throttle aggressively

---

## License

MIT
