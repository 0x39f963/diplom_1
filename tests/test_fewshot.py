from __future__ import annotations

from eva_agent.nlu.fewshot import retrieve_examples
from eva_agent.settings import settings


def test_retrieve_examples_finds_party_cases_in_bm25_only(monkeypatch) -> None:
    monkeypatch.setattr(settings, "fewshot_dense_enabled", False)

    examples = retrieve_examples("покажи стороны договора CT-1", k=5)

    assert examples
    assert any(
        example.id.startswith("data-parties")
        or "eva_get_contract_parties" in example.tags
        for example in examples
    )


def test_retrieve_examples_bm25_only_does_not_load_embedder(monkeypatch) -> None:
    import eva_agent.nlu.fewshot as fewshot_module

    monkeypatch.setattr(settings, "fewshot_dense_enabled", False)

    def fail_embedder():
        raise AssertionError("dense embedder should not load")

    monkeypatch.setattr(fewshot_module, "_embedder", fail_embedder)

    assert retrieve_examples("какие документы по договору CT-1", k=3)


def test_retrieve_examples_merges_dense_rank_with_rrf(monkeypatch) -> None:
    import eva_agent.nlu.fewshot as fewshot_module

    examples, _ = fewshot_module._sparse_index()
    dense_target = next(index for index, example in enumerate(examples) if example.id == "data-creative-1")

    def fake_dense_rank(query: str, indexed_examples):
        del query, indexed_examples
        return [dense_target]

    monkeypatch.setattr(settings, "fewshot_dense_enabled", True)
    monkeypatch.setattr(fewshot_module, "_dense_rank", fake_dense_rank)

    result = retrieve_examples("покажи стороны договора CT-1", k=5)

    assert any(example.id == "data-creative-1" for example in result)
