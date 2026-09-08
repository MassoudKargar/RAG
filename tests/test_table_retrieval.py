"""Phase 10 — Table (balance sheet / cash flow) retrieval tests.

Requires the live embedding service (127.0.0.1:9010) and the production
ChromaDB corpus (5 MSFT 10-K filings). Skipped when the embedder is down.
"""
import os
import re
import sys
import socket

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def _embedding_service_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 9010), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _embedding_service_up(),
    reason="local embedding service (127.0.0.1:9010) not reachable",
)


def _rag():
    from app.config import settings as settings_module
    live_url = "http://127.0.0.1:9010/embed"
    os.environ["LOCAL_EMBEDDING_API_URL"] = live_url
    settings_module.settings.LOCAL_EMBEDDING_API_URL = live_url
    settings_module.settings.CHROMA_PERSIST_DIRECTORY = "/var/rag_app/chroma_db"
    from app.services.core.rag_service import RAGService
    from app.services.providers.local_embedding_service import LocalEmbeddingProvider
    svc = RAGService()
    svc._embedding_provider = LocalEmbeddingProvider()
    return svc


def _norm(s):
    return s.replace("$", "").replace(",", "").replace(" ", "").replace("\u00a0", "").replace("'", "")


class TestBalanceSheetQueries:
    """Ground-truth values from the actual corpus (verified against DB)."""

    CASES = [
        ("What were Microsoft's total assets in fiscal year 2024?", "512163"),
        ("Microsoft cash and cash equivalents fiscal year 2024", "18315"),
        ("Microsoft total stockholders equity fiscal year 2024", "268477"),
        ("Microsoft long-term debt fiscal year 2024", "42688"),
        ("Microsoft goodwill fiscal year 2026", "119651"),
        ("Microsoft total liabilities fiscal year 2024", "243686"),
        ("What were Microsoft's total assets in fiscal year 2026?", "758376"),
        ("Net cash from operations fiscal year 2024", "118548"),
    ]

    @pytest.mark.parametrize("q,val", CASES, ids=[c[0][:30] for c in CASES])
    def test_value_retrieved(self, q, val):
        rag = _rag()
        emb = rag.embedding_provider.create_embedding(q)
        res = rag.search_similar_documents(emb, limit=8, query=q, where=rag.year_filter(q))
        docs = res["documents"][0]
        found = next((i for i, d in enumerate(docs) if d and _norm(val) in _norm(d)), None)
        assert found is not None, f"{q}: value {val} not in top-8\n" + "\n".join(
            f"  #{i}: {_norm(d)[:60]}" for i, d in enumerate(docs[:3])
        )
        assert found <= 3, f"{q}: value at rank {found} (want <= 3)"

    def test_metric_labels_mapped(self):
        """The canonical metric labels exist for every added table concept."""
        from app.services.core.query_analyzer import query_analyzer, metric_label_regexes
        for q, _ in self.CASES:
            m = query_analyzer.extract_metric(q)
            assert m is not None, f"no metric for {q}"
            assert metric_label_regexes(m), f"no label regexes for {m}"