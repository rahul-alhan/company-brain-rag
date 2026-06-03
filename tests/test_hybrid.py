"""RRF unit tests — no faiss / openai / live keys required."""
from retrieval.hybrid import HybridRetriever, rrf_fuse


def test_rrf_fuses_lists_by_inverse_rank():
    sparse_ids = ["a", "b", "c"]
    dense_ids = ["b", "d", "a"]
    ranked = rrf_fuse(sparse_ids, dense_ids)
    # Docs that appear in BOTH lists should win
    top_two = {ranked[0][0], ranked[1][0]}
    assert top_two == {"a", "b"}


def test_rrf_handles_empty_inputs():
    assert rrf_fuse([], []) == []
    assert rrf_fuse(["a"], []) == [("a", 1.0 / 61)]


def test_rrf_is_order_sensitive():
    """Earlier rank → higher contribution to the fused score."""
    ranked = rrf_fuse(["x", "y"])
    assert ranked[0][0] == "x"
    assert ranked[0][1] > ranked[1][1]


def test_hybrid_module_imports_without_faiss():
    """Class exists at import time — instantiation still needs faiss/openai."""
    assert HybridRetriever is not None
    assert hasattr(HybridRetriever, "search")
