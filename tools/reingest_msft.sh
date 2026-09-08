#!/bin/bash
# Re-ingest the 5 MSFT 10-K filings with the updated chunker (v4 text,
# section/page metadata). Uses direct RAGService.add_document with the SAME
# document_ids -> deterministic chunk ids -> upsert replaces old chunks.
set -e
cd /var/rag_app
export HOME=/root

.venv/bin/python - <<'EOF'
import time, sys
sys.path.insert(0, "/var/rag_app")
from app.services.core.rag_service import rag_service

FILES = {
    2022: "/tmp/msft_fy2022_v4.txt",
    2023: "/tmp/msft_fy2023_v4.txt",
    2024: "/tmp/msft_fy2024_v4.txt",
    2025: "/tmp/msft_fy2025_v4.txt",
    2026: "/tmp/msft_fy2026_v4.txt",
}
for year in sorted(FILES):
    with open(FILES[year], encoding="utf-8") as f:
        text = f.read()
    doc_id = f"microsoft-10k-{year}"
    t0 = time.time()
    result = rag_service.add_document(
        text,
        metadata={
            "company": "Microsoft",
            "fiscal_year": year,
            "document_type": "10-K",
            "document_id": doc_id,
        },
        document_id=doc_id,
        source=f"10-K FY{year}",
    )
    print(f"FY{year}: {result['chunks_added']} chunks in {time.time()-t0:.0f}s", flush=True)

print("REINGEST DONE")
# verify final count
from app.services.core.vector_store import VectorStoreService
col = VectorStoreService().get_collection("RAG_COLLECTION")
print("collection count:", col.count())
EOF