"""Phase 9 — Section-aware retrieval tests.

Covers:
- correct section detection (EN + FA)
- section ranking boost (query asks about a section -> that section ranks top)
- irrelevant section demotion (query NOT about a section doesn't boost any)
- Persian section queries
- year + metric + section combinations

Requires the live embedding service (127.0.0.1:9010); skipped when down.
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
    """RAG service bound to the PRODUCTION corpus + live embedder on 9010.

    conftest.py redirects CHROMA_PERSIST_DIRECTORY to a temp dir and the
    settings singleton to the legacy 8010 port; here we force the live
    embedding URL and rebuild the vector store against the production DB so
    the section ranking tests run on real 10-K chunks.
    """
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


class TestSectionDetection:
    @pytest.fixture(autouse=True)
    def setup(self):
        from app.services.core.query_analyzer import query_analyzer
        self.qa = query_analyzer

    def test_english_sections(self):
        cases = {
            "What does Microsoft say about risk factors?": "Item 1A",
            "Management's Discussion and Analysis of financial condition": "Item 7",
            "cybersecurity risks at Microsoft": "Item 1C",
            "business overview of Microsoft": "Item 1",
            "Which financial statements show net income?": "Item 8",
            "controls and procedures at Microsoft": "Item 9A",
            "corporate governance and directors": "Item 10",
        }
        for q, expected in cases.items():
            assert self.qa.extract_section(q) == expected, q

    def test_persian_sections(self):
        cases = {
            "مخاطرات اصلی کسبوکار مایکروسافت چیست؟": "Item 1A",
            "صورت‌های مالی مایکروسافت": "Item 8",
            "حاکمیت شرکتی مایکروسافت": "Item 10",
            "امنیت سایبری مایکروسافت": "Item 1C",
        }
        for q, expected in cases.items():
            assert self.qa.extract_section(q) == expected, q

    def test_no_section_for_financial(self):
        assert self.qa.extract_section("What was Microsoft's revenue in fiscal year 2024?") is None


class TestSectionRanking:
    def test_risk_factors_ranks_1A(self):
        rag = _rag()
        q = "What does Microsoft say about risk factors?"
        emb = rag.embedding_provider.create_embedding(q)
        res = rag.search_similar_documents(emb, limit=5, query=q, where=rag.year_filter(q))
        metas = res["metadatas"][0]
        canon = [rag._canonical_section((m or {}).get("section")) for m in metas]
        assert "Item 1A" in canon[:3], canon

    def test_business_overview_ranks_1(self):
        rag = _rag()
        q = "Microsoft's business overview"
        emb = rag.embedding_provider.create_embedding(q)
        res = rag.search_similar_documents(emb, limit=5, query=q, where=rag.year_filter(q))
        metas = res["metadatas"][0]
        canon = [rag._canonical_section((m or {}).get("section")) for m in metas]
        assert "Item 1" in canon[:3], canon

    def test_revenue_query_not_dominated_by_section(self):
        """A metric query must NOT be hijacked by section boost."""
        rag = _rag()
        q = "Microsoft net income fiscal year 2024"
        emb = rag.embedding_provider.create_embedding(q)
        res = rag.search_similar_documents(emb, limit=5, query=q, where=rag.year_filter(q))
        docs = res["documents"][0]
        def norm(s): return s.replace("$", "").replace(",", "").replace(" ", "").replace("\u00a0", "")
        assert any("88136" in norm(d) for d in docs[:3])

    def test_year_metric_section_combined(self):
        rag = _rag()
        q = "What risks did Microsoft highlight in fiscal year 2024?"
        emb = rag.embedding_provider.create_embedding(q)
        res = rag.search_similar_documents(emb, limit=5, query=q, where=rag.year_filter(q))
        metas = res["metadatas"][0]
        canon = [rag._canonical_section((m or {}).get("section")) for m in metas]
        assert any(c in ("Item 1A", "Item 7") for c in canon[:3]), canon