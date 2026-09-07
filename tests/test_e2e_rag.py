"""End-to-end RAG flow test with a large synthetic Persian document.

Requires the local embedding service (127.0.0.1:9010) to be running.
Skipped automatically when the service is unreachable.

Flow under test:
    Document -> Chunk -> Embedding -> ChromaDB -> Search -> Context
"""
import os
import sys
import uuid
import socket

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def _embedding_service_up() -> bool:
    """Check whether the local embedding service is reachable."""
    try:
        with socket.create_connection(("127.0.0.1", 9010), timeout=2):
            return True
    except OSError:
        return False


# Isolate from production data: temp ChromaDB dir
pytestmark = pytest.mark.skipif(
    not _embedding_service_up(),
    reason="local embedding service (127.0.0.1:9010) not reachable",
)


def _make_test_document(lines: int = 5000) -> str:
    """Build a synthetic Persian document with identifiable facts at known positions."""
    facts = {
        50: ("TEST-001", "12345", "قانون اول"),
        1500: ("TEST-002", "67890", "قانون دوم"),
        3000: ("TEST-003", "24680", "قانون سوم"),
        4500: ("TEST-004", "13579", "قانون چهارم"),
    }
    sections = [
        "فصل اول: قوانین پایه",
        "فصل دوم: قوانین اجرایی",
        "فصل سوم: قوانین مالی",
        "فصل چهارم: قوانین تکمیلی",
    ]
    out_lines = []
    section_idx = 0
    for i in range(1, lines + 1):
        if i % 1200 == 1:
            section_idx = min(section_idx + (1 if i > 1 else 0), len(sections) - 1)
            out_lines.append(f"# {sections[section_idx]}")
            out_lines.append("")
        if i in facts:
            _, value, law = facts[i]
            out_lines.append(f"شناسه تست {facts[i][0]} مربوط به {law} است و مقدار آن {value} می‌باشد.")
        else:
            out_lines.append(
                f"بند {i}: طبق مقررات جاری، کلیه موارد اداری باید مطابق ضوابط مصوب {i} تنظیم شود."
            )
        out_lines.append("")
    return "\n".join(out_lines)


@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    """RAG service bound to a temp ChromaDB dir + the live embedding service."""
    # The production embedding microservice runs on 127.0.0.1:9010, while
    # conftest.py forces the legacy 8010 port (not running). Override the
    # cached settings singleton directly so both the vector store and the
    # embedding provider hit the live service.
    from app.config import settings as settings_module

    live_url = "http://127.0.0.1:9010/embed"
    os.environ["LOCAL_EMBEDDING_API_URL"] = live_url
    settings_module.settings.LOCAL_EMBEDDING_API_URL = live_url

    tmp_db = tmp_path_factory.mktemp("chroma_e2e")
    settings_module.settings.CHROMA_PERSIST_DIRECTORY = str(tmp_db)
    # Force re-resolution of the vector store client with the temp dir
    from app.services.core.rag_service import RAGService

    service = RAGService()
    service.collection_name = f"e2e_{uuid.uuid4().hex[:8]}"
    service.initialize_collection()
    yield service


def test_large_document_ingestion_and_retrieval(rag):
    """Insert a 5000-line document, verify chunks + retrieval of distant facts."""
    doc = _make_test_document(5000)
    result = rag.add_document(doc, metadata={"source": "laws.txt", "topic": "قوانین"})
    assert result["chunks_added"] >= 10, result

    # Verify deterministic, unique chunk ids
    assert len(result["chunk_ids"]) == len(set(result["chunk_ids"])) > 0

    # Search for the fact near the END of the document (TEST-004)
    embedding = rag.embedding_provider.create_embedding("مقدار TEST-004 چیست؟")
    search = rag.search_similar_documents(embedding, limit=10)
    docs = (search.get("documents") or [[]])[0]
    metas = (search.get("metadatas") or [[]])[0]
    distances = search.get("distances") or [[]]

    assert docs, "no documents retrieved"
    assert len(distances[0]) == len(docs)

    # The retrieved context must contain the TEST-004 fact
    context = rag.build_context(search, limit=5)
    assert "TEST-004" in context, "fact near the END of the document was not retrieved"
    assert "13579" in context

    # Metadata preserved
    meta0 = metas[0] or {}
    assert meta0.get("document_id"), meta0
    assert "chunk_index" in meta0
    assert meta0.get("total_chunks", 0) > 0
    assert meta0.get("source") == "laws.txt"
    assert meta0.get("topic") == "قوانین"


def test_beginning_and_middle_retrieval(rag):
    """Facts near the beginning/middle of the document must also be retrievable."""
    for marker, value in [("TEST-001", "12345"), ("TEST-002", "67890"), ("TEST-003", "24680")]:
        embedding = rag.embedding_provider.create_embedding(f"مقدار {marker} چیست؟")
        search = rag.search_similar_documents(embedding, limit=10)
        context = rag.build_context(search, limit=5)
        assert marker in context, f"{marker} not retrieved"
        assert value in context, f"value for {marker} not retrieved"


def test_reinsert_same_document_replaces(rag):
    """Re-adding with the same document_id must not duplicate chunks."""
    doc = _make_test_document(5000)
    first = rag.add_document(doc, metadata={"source": "laws.txt", "document_id": "fixed_doc"})
    count_before = rag.vector_store.get_collection(rag.collection_name).count()

    second = rag.add_document(doc, metadata={"source": "laws.txt", "document_id": "fixed_doc"})
    count_after = rag.vector_store.get_collection(rag.collection_name).count()

    assert first["chunks_added"] == second["chunks_added"]
    assert count_after <= count_before + 1, f"duplicated chunks: {count_before} -> {count_after}"