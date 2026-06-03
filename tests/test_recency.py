from datetime import datetime, timezone, timedelta

from retrieval.types import RetrievedChunk
from retrieval.recency import recency_rerank


def _rc(chunk_id, source, days_old, score=1.0):
    ts = datetime.now(timezone.utc) - timedelta(days=days_old)
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        chunk={"source": source, "timestamp_iso": ts.isoformat()},
    )


def test_recent_slack_outranks_old_slack_at_same_score():
    chunks = [_rc("old", "slack", days_old=120), _rc("new", "slack", days_old=1)]
    out = recency_rerank(chunks, half_life_days={"slack": 30})
    assert out[0].chunk_id == "new"


def test_notion_decays_slower_than_slack():
    # Same age, same starting score — slack should decay more
    slack = _rc("s", "slack", days_old=60)
    notion = _rc("n", "notion", days_old=60)
    out = recency_rerank([slack, notion], half_life_days={"slack": 30, "notion": 180})
    assert out[0].chunk_id == "n"


def test_higher_base_score_can_still_lose_if_much_older():
    fresh_weak = _rc("fresh", "slack", days_old=1, score=0.5)
    stale_strong = _rc("stale", "slack", days_old=180, score=1.0)
    out = recency_rerank([fresh_weak, stale_strong], half_life_days={"slack": 30})
    # 1.0 * 0.5^6 = 0.0156 vs 0.5 * ~1 = ~0.5
    assert out[0].chunk_id == "fresh"
