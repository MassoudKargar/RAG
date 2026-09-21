"""Regression tests for safe normalization of Chroma search results.

The public /v1/vector_db/search_documents endpoint must never fail validation
when Chroma returns null/empty document entries (e.g. short or hyphenated
queries). Invalid entries are skipped while preserving index alignment;
all-invalid responses are HTTP 200 with an empty list.
"""
import pytest
from app.routes.vector_db import normalize_search_results


def _ids(n):
    return [f"chunk_{i:05d}" for i in range(n)]


def test_short_query_returns_empty_when_all_null():
    out = normalize_search_results(ids=_ids(2), documents=[None, None], metadatas=[{}, {}], distances=[0.1, 0.2])
    assert out == []


def test_hyphenated_query_null_mixed_with_valid():
    out = normalize_search_results(
        ids=_ids(3), documents=[None, "valid chunk text A", "valid chunk text B"],
        metadatas=[{}, {"document_id": "d1"}, {"document_id": "d2"}],
        distances=[0.1, 0.2, 0.3],
    )
    assert len(out) == 2
    assert out[0].id == 1 and out[0].text == "valid chunk text A"
    assert out[1].id == 2 and out[1].text == "valid chunk text B"


def test_zenith_abc_query():
    """ZENITH-abc style marker queries caused text=None 502s before the fix."""
    out = normalize_search_results(ids=_ids(1), documents=[None], metadatas=[{}], distances=[0.4])
    assert out == []


def test_full_marker_phrase_returns_valid():
    out = normalize_search_results(
        ids=_ids(1), documents=["the unique phrase ZENITH-9c1df9ba appears here"],
        metadatas=[{"document_id": "d1", "source": "z.txt"}], distances=[0.05],
    )
    assert len(out) == 1
    assert "ZENITH-9c1df9ba" in out[0].text
    assert out[0].metadata["document_id"] == "d1"


def test_empty_string_mixed_with_valid():
    out = normalize_search_results(
        ids=_ids(2), documents=["", "real text"], metadatas=[{}, {"document_id": "d"}], distances=[0.1, 0.2]
    )
    assert len(out) == 1 and out[0].text == "real text"


def test_missing_metadata_kept():
    out = normalize_search_results(ids=_ids(1), documents=["text"], metadatas=[None], distances=[0.1])
    assert len(out) == 1 and out[0].metadata is None


def test_all_null_response_empty():
    out = normalize_search_results(ids=_ids(3), documents=[None, None, None], metadatas=[None, None, None], distances=[0.1, 0.2, 0.3])
    assert out == []


def test_no_result_response_empty():
    out = normalize_search_results(ids=[], documents=[], metadatas=[], distances=[])
    assert out == []


def test_missing_distances_column():
    out = normalize_search_results(ids=_ids(1), documents=["text"], metadatas=[{"document_id": "d"}], distances=None)
    assert len(out) == 1 and out[0].score is None


def test_missing_columns():
    out = normalize_search_results(ids=None, documents=None, metadatas=None, distances=None)
    assert out == []


def test_existing_production_query_shape():
    ids = _ids(2)
    docs = [
        "years: (In millions, except percentages and per share amounts) 2024 2023 Percent",
        "Microsoft net income fiscal year 2024",
    ]
    metas = [{"document_id": "microsoft-10k-2024", "fiscal_year": 2024}, {"document_id": "microsoft-10k-2024", "fiscal_year": 2024}]
    dists = [0.18, 0.34]
    out = normalize_search_results(ids=ids, documents=docs, metadatas=metas, distances=dists)
    assert len(out) == 2
    assert out[0].score == 0.18 and out[1].score == 0.34
    assert out[0].metadata["fiscal_year"] == 2024


def test_never_converts_none_to_string():
    out = normalize_search_results(ids=_ids(1), documents=[None], metadatas=[{}], distances=[0.9])
    # The literal string "None" must never appear as a chunk.
    assert all(o.text != "None" for o in out)