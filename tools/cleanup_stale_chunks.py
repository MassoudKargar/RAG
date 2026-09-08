#!/usr/bin/env python3
"""Remove stale chunks from the live collection after a chunker change.

Re-chunks each of the 5 MSFT docs (deterministic ids), compares against the
ids currently stored in ChromaDB, and deletes only the ids that are no longer
produced (chunk ids removed/renumbered by the new chunker). New/unchanged ids
are left alone so embedding work is not repeated.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.core.chunker import ChunkerService

FILES = {
    2022: "/tmp/msft_fy2022_v4.txt",
    2023: "/tmp/msft_fy2023_v4.txt",
    2024: "/tmp/msft_fy2024_v4.txt",
    2025: "/tmp/msft_fy2025_v4.txt",
    2026: "/tmp/msft_fy2026_v4.txt",
}

def main():
    chunker = ChunkerService(chunk_size=800, chunk_overlap=100)
    from app.services.core.vector_store import VectorStoreService
    vs = VectorStoreService()
    col = vs.get_collection("RAG_COLLECTION")
    total_deleted = 0
    for year in sorted(FILES):
        doc_id = f"microsoft-10k-{year}"
        with open(FILES[year], encoding="utf-8") as f:
            text = f.read()
        chunks = chunker.chunk_document(
            text,
            document_id=doc_id,
            metadata={"company": "Microsoft", "fiscal_year": year, "document_id": doc_id},
            source=f"10-K FY{year}",
        )
        expected_ids = {c.id for c in chunks}
        existing = col.get(where={"document_id": doc_id}, include=[])["ids"]
        stale = [i for i in existing if i not in expected_ids]
        if stale:
            # delete in batches of 500
            for start in range(0, len(stale), 500):
                col.delete(ids=stale[start:start + 500])
            total_deleted += len(stale)
            print(f"{doc_id}: deleted {len(stale)} stale chunks ({len(existing)} -> {len(expected_ids)})")
        else:
            print(f"{doc_id}: no stale chunks ({len(existing)})")
    print("total deleted:", total_deleted)
    print("final count:", col.count())
    # Also ensure all expected ids exist (safety)
    missing = 0
    for year in sorted(FILES):
        doc_id = f"microsoft-10k-{year}"
        with open(FILES[year], encoding="utf-8") as f:
            text = f.read()
        expected_ids = {c.id for c in chunker.chunk_document(
            text, document_id=doc_id,
            metadata={"company": "Microsoft", "fiscal_year": year, "document_id": doc_id},
            source=f"10-K FY{year}",
        )}
        existing = set(col.get(where={"document_id": doc_id}, include=[])["ids"])
        m = expected_ids - existing
        if m:
            missing += len(m)
            print(f"WARNING: {doc_id} missing {len(m)} expected chunks")
    print("missing expected:", missing)


if __name__ == "__main__":
    main()