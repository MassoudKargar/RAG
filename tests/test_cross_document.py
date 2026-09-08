"""Phase 13 — Cross-document retrieval tests.

Queries whose correct answer requires values from multiple Microsoft 10-K
filings (different fiscal years). Verifies every needed year's value is
retrievable in one shot.
"""
import os
import sys
import socket

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def _up():
    try:
        with socket.create_connection(("127.0.0.1", 9010), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _up(), reason="embedder down")


def _rag():
    from app.config import settings as s
    live = "http://127.0.0.1:9010/embed"
    os.environ["LOCAL_EMBEDDING_API_URL"] = live
    s.settings.LOCAL_EMBEDDING_API_URL = live
    s.settings.CHROMA_PERSIST_DIRECTORY = "/var/rag_app/chroma_db"
    from app.services.core.rag_service import RAGService
    from app.services.providers.local_embedding_service import LocalEmbeddingProvider
    svc = RAGService()
    svc._embedding_provider = LocalEmbeddingProvider()
    return svc


def _norm(x):
    return x.replace("$", "").replace(",", "").replace(" ", "").replace("\u00a0", "")


REVENUE = {2022: 198270, 2023: 211915, 2024: 245122, 2025: 281724, 2026: 331839}
NET_INCOME = {2022: 72738, 2023: 72361, 2024: 88136, 2025: 101832, 2026: 133749}


class TestCrossDocument:
    CASES = [
        ("How did Microsoft's total revenue compare in fiscal year 2022 vs 2026?",
         [2022, 2026], REVENUE),
        ("Compare Microsoft's net income for fiscal 2023 and fiscal 2025",
         [2023, 2025], NET_INCOME),
        ("What was the trend of Microsoft revenue from 2022 to 2024?",
         [2022, 2023, 2024], REVENUE),
        ("Microsoft revenue 2024 vs 2026 compared to net income 2024",
         [2024, 2026], REVENUE),
    ]

    @pytest.mark.parametrize("q,years,data", CASES, ids=[c[0][:28] for c in CASES])
    def test_multi_doc_values_retrievable(self, q, years, data):
        rag = _rag()
        emb = rag.embedding_provider.create_embedding(q)
        res = rag.search_similar_documents(emb, limit=12, query=q, where=rag.year_filter(q))
        docs = (res.get("documents") or [[]])[0]
        # every requested year's table value must be present across the pool
        missing = [y for y in years if not any(
            _norm(str(data[y])) in _norm(d) for d in docs)]
        assert not missing, f"{q}: missing values for {missing}"