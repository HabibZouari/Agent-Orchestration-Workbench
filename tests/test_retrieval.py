import pytest

from agent_orchestration.retrieval import (
    RRF_K,
    SAMPLE_DOCUMENTS,
    SQLiteStore,
    query_terms,
    reciprocal_rank_fusion,
)
from agent_orchestration.schemas import DocumentInput


def test_rrf_rewards_documents_ranked_by_both_retrievers() -> None:
    scores = reciprocal_rank_fusion(["a", "b"], ["b", "c"])

    assert scores["b"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert scores["a"] == pytest.approx(1 / (RRF_K + 1))
    assert max(scores, key=scores.get) == "b"


def test_query_terms_strip_query_syntax() -> None:
    assert query_terms('Policy "OR" data-retention* policy') == [
        "policy",
        "or",
        "data",
        "retention",
    ]


def test_fresh_store_seeds_samples_once(tmp_path, embedder) -> None:
    path = tmp_path / "docs.db"
    SQLiteStore(path, embedder)
    reopened = SQLiteStore(path, embedder)

    ids = [doc.id for doc in reopened.list_documents()]
    assert sorted(ids) == sorted(doc.id for doc in SAMPLE_DOCUMENTS)


def test_seeding_can_be_disabled(tmp_path, embedder) -> None:
    assert SQLiteStore(tmp_path / "docs.db", embedder, seed=False).list_documents() == []


def test_search_finds_seeded_policy(store) -> None:
    results = store.search("data retention policy", top_k=3)

    assert results[0].id == "demo-policy"
    assert results[0].metadata["category"] == "compliance"
    assert results[0].score > results[-1].score


def test_upsert_keeps_metadata_and_replaces_content(store) -> None:
    store.upsert([DocumentInput(id="faq", content="Original zebra text", metadata={"v": 1})])
    store.upsert([DocumentInput(id="faq", content="Updated giraffe text", metadata={"v": 2})])

    hit = store.search("giraffe", top_k=1)[0]
    assert (hit.id, hit.content, hit.metadata) == ("faq", "Updated giraffe text", {"v": 2})
    assert "faq" not in store._lexical_ids(["zebra"])


def test_delete_removes_document_from_both_indexes(store) -> None:
    assert store.delete("demo-policy") is True
    assert store.delete("demo-policy") is False

    assert "demo-policy" not in {doc.id for doc in store.list_documents()}
    assert "demo-policy" not in {r.id for r in store.search("data retention policy", top_k=10)}


def test_search_on_empty_store_returns_nothing(tmp_path, embedder) -> None:
    assert SQLiteStore(tmp_path / "docs.db", embedder, seed=False).search("x") == []
