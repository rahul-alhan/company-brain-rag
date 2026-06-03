from retrieval.acl_filter import principals_for_user, filter_by_principals
from retrieval.types import RetrievedChunk


GROUPS = {
    "team:engineering": ["alice@acme.com", "bob@acme.com"],
    "team:security": ["alice@acme.com"],
}


def _rc(chunk_id, acl):
    return RetrievedChunk(chunk_id=chunk_id, score=1.0, chunk={"acl_principals": acl})


def test_principals_include_user_and_groups():
    p = principals_for_user("alice@acme.com", GROUPS)
    assert "user:alice@acme.com" in p
    assert "team:engineering" in p
    assert "team:security" in p
    assert "group:all-employees" in p


def test_outsider_does_not_get_team_principals():
    p = principals_for_user("contractor@acme.com", GROUPS)
    assert "team:engineering" not in p
    assert "team:security" not in p


def test_filter_drops_chunks_user_cannot_see():
    p = principals_for_user("contractor@acme.com", GROUPS)
    candidates = [
        _rc("c1", ["team:security"]),
        _rc("c2", ["group:all-employees"]),
        _rc("c3", ["team:engineering"]),
    ]
    out = filter_by_principals(candidates, p, k_final=10)
    assert [c.chunk_id for c in out] == ["c2"]


def test_filter_allows_chunks_with_intersecting_principal():
    p = principals_for_user("alice@acme.com", GROUPS)
    candidates = [
        _rc("c1", ["team:security"]),
        _rc("c2", ["user:bob@acme.com"]),
        _rc("c3", ["team:engineering", "team:security"]),
    ]
    out = filter_by_principals(candidates, p, k_final=10)
    assert {c.chunk_id for c in out} == {"c1", "c3"}


def test_empty_acl_treated_as_public():
    p = principals_for_user("contractor@acme.com", GROUPS)
    candidates = [_rc("c1", [])]
    out = filter_by_principals(candidates, p, k_final=10)
    assert len(out) == 1
