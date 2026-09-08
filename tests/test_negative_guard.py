"""Phase 12 — Negative / hallucination-guard tests.

Verifies the honest no-answer path:
- out-of-corpus fiscal years (2015/1999/2010/1995) -> empty result
- other-company queries (Apple/Amazon/Google) -> empty result (corpus is
  Microsoft-only; prevents hallucinating MSFT figures)
- Microsoft + Persian queries still retrieve correctly

Requires the live embedding service and production corpus.
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
    not _embedding_service_up(), reason="embedder down",
)


def _rag():
    from app.config import settings as settings_module
    live = "http://127.0.0.1:9010/embed"
    os.environ["LOCAL_EMBEDDING_API_URL"] = live
    settings_module.settings.LOCAL_EMBEDDING_API_URL = live
    settings_module.settings.CHROMA_PERSIST_DIRECTORY = "/var/rag_app/chroma_db"
    from app.services.core.rag_service import RAGService
    from app.services.providers.local_embedding_service import LocalEmbeddingProvider
    svc = RAGService()
    svc._embedding_provider = LocalEmbeddingProvider()
    return svc


def _norm(s):
    return s.replace("$", "").replace(",", "").replace(" ", "").replace("\u00a0", "")


class TestNegativeGuard:
    @pytest.mark.parametrize("q", [
        "What was Microsoft's revenue in fiscal year 2015?",
        "What was Microsoft's net income in fiscal year 1999?",
        "Microsoft's total headcount in 1995",
        "What was Microsoft's debt in fiscal 2010?",
    ])
    def test_out_of_corpus_year_empty(self, q):
        rag = _rag()
        emb = rag.embedding_provider.create_embedding(q)
        res = rag.search_similar_documents(emb, limit=5, query=q, where=rag.year_filter(q))
        assert len((res.get("documents") or [[]])[0]) == 0, q

    @pytest.mark.parametrize("q", [
        "Apple's revenue in fiscal year 2024",
        "What was Amazon's net income in 2023?",
        "What was Google's revenue in 2024?",
    ])
    def test_other_company_empty(self, q):
        rag = _rag()
        emb = rag.embedding_provider.create_embedding(q)
        res = rag.search_similar_documents(emb, limit=5, query=q, where=rag.year_filter(q))
        assert len((res.get("documents") or [[]])[0]) == 0, q

    @pytest.mark.parametrize("q,val", [
        ("Microsoft net income fiscal year 2024", "88136"),
        ("What was Microsoft's total revenue in fiscal year 2025?", "281724"),
        ("مایکروسافت در سال مالی ۲۰۲۵ چه مقدار درآمد داشت؟", "281724"),
    ])
    def test_microsoft_still_works(self, q, val):
        rag = _rag()
        emb = rag.embedding_provider.create_embedding(q)
        res = rag.search_similar_documents(emb, limit=5, query=q, where=rag.year_filter(q))
        docs = (res.get("documents") or [[]])[0]
        assert any(_norm(val) in _norm(d) for d in docs), q