"""Few-shot retrieval over the mixed benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from eva_agent.nlu.preprocess import preprocess
from eva_agent.settings import settings

_RRF_K = 60
_INDEX_DIR = Path(__file__).with_name("fewshot_index")
_META_PATH = _INDEX_DIR / "meta.json"
_EMBEDDINGS_PATH = _INDEX_DIR / "embeddings.npy"


class Example(BaseModel):
    id: str
    query: str
    lemmas: list[str] = Field(default_factory=list)
    intent: str = ""
    route: str = ""
    tags: list[str] = Field(default_factory=list)
    score: float = 0.0


def retrieve_examples(query: str, k: int = 5) -> list[Example]:
    examples, bm25 = _sparse_index()
    if not examples:
        return []

    query_lemmas = preprocess(query).lemmas
    sparse_scores = bm25.get_scores(query_lemmas) if query_lemmas else [0.0] * len(examples)
    sparse_rank = _rank(sparse_scores)
    rankings = [sparse_rank]

    if settings.fewshot_dense_enabled:
        dense_rank = _dense_rank(query, examples)
        if dense_rank:
            rankings.append(dense_rank)

    fused = _rrf(rankings)
    ordered = sorted(fused, key=lambda index: fused[index], reverse=True)[:k]
    return [examples[index].model_copy(update={"score": fused[index]}) for index in ordered]


def build_index(*, rebuild: bool = False) -> Path:
    examples = _examples()
    _ensure_dense_index(examples, rebuild=rebuild)
    return _INDEX_DIR


@lru_cache(maxsize=1)
def _examples() -> tuple[Example, ...]:
    path = _repo_root() / "bench" / "_mixed_109.jsonl"
    rows: list[Example] = []
    if not path.exists():
        return tuple()
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        query = str(raw.get("input") or raw.get("query") or "")
        if not query:
            continue
        intent = str(raw.get("expected_intent") or "")
        route = str(raw.get("gold_route") or "")
        rows.append(
            Example(
                id=str(raw.get("id") or f"example-{len(rows) + 1}"),
                query=query,
                lemmas=preprocess(query).lemmas,
                intent=intent,
                route=route,
                tags=_tags(raw, intent, route),
            )
        )
    return tuple(rows)


@lru_cache(maxsize=1)
def _sparse_index() -> tuple[tuple[Example, ...], BM25Okapi]:
    examples = _examples()
    corpus = [example.lemmas for example in examples]
    return examples, BM25Okapi(corpus)


def _tags(raw: dict[str, Any], intent: str, route: str) -> list[str]:
    tags = [value for value in (intent, route) if value]
    if raw.get("clarify_warranted") is True:
        tags.append("clarify")
    asserts = raw.get("asserts")
    if isinstance(asserts, list):
        for item in asserts:
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            if isinstance(value, str):
                tags.append(value)
            elif isinstance(value, list):
                tags.extend(str(part) for part in value)
    return _unique(tags)


def _dense_rank(query: str, examples: tuple[Example, ...]) -> list[int]:
    try:
        import numpy as np

        embeddings = _ensure_dense_index(examples, rebuild=False)
        query_embedding = _embed([query])
        scores = np.matmul(embeddings, query_embedding[0])
    except Exception:
        return []
    return _rank(scores)


def _ensure_dense_index(examples: tuple[Example, ...], *, rebuild: bool) -> Any:
    import numpy as np

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    expected = _fingerprint(examples)
    if not rebuild and _META_PATH.exists() and _EMBEDDINGS_PATH.exists():
        try:
            meta = json.loads(_META_PATH.read_text("utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if meta.get("fingerprint") == expected:
            return np.load(_EMBEDDINGS_PATH)

    embeddings = _embed([example.query for example in examples])
    np.save(_EMBEDDINGS_PATH, embeddings)
    _META_PATH.write_text(
        json.dumps(
            {
                "fingerprint": expected,
                "model": settings.fewshot_embed_model,
                "device": settings.fewshot_embed_device,
                "count": len(examples),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return embeddings


@lru_cache(maxsize=1)
def _embedder() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.fewshot_embed_model, device=settings.fewshot_embed_device)


def _embed(texts: list[str]) -> Any:
    return _embedder().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def _rank(scores: Any) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)


def _rrf(rankings: list[list[int]]) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (_RRF_K + rank)
    return fused


def _fingerprint(examples: tuple[Example, ...]) -> str:
    payload = {
        "model": settings.fewshot_embed_model,
        "examples": [
            {
                "id": example.id,
                "query": example.query,
                "intent": example.intent,
                "route": example.route,
            }
            for example in examples
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).parents[3]


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    path = build_index(rebuild=args.rebuild)
    print(path)


if __name__ == "__main__":
    _main()
