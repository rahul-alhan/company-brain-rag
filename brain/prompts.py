SYSTEM = """You are the company knowledge assistant.

Rules you must follow:
1. Answer ONLY from the provided <context> chunks. If the context is insufficient, say so.
2. Every factual claim must be followed by a citation in the form (source: <source>::<chunk_id>, last_edited: <iso>).
3. If two sources disagree, surface the disagreement and cite both.
4. Prefer the most recent source when claims are time-sensitive.
5. Never fabricate document titles, authors, or dates.
"""

USER_TEMPLATE = """<context>
{context}
</context>

Question: {query}
"""


def format_context(chunks) -> str:
    blocks = []
    for c in chunks:
        meta = (
            f"[source: {c.chunk['source']}::{c.chunk['chunk_id']}, "
            f"title: {c.chunk['title']!r}, "
            f"last_edited: {c.chunk['timestamp_iso']}]"
        )
        blocks.append(f"{meta}\n{c.chunk['text']}\n")
    return "\n---\n".join(blocks)
