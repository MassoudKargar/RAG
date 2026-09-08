#!/usr/bin/env python3
"""PHASE 1 — Chunking integrity audit on the real Microsoft 10-K corpus.

Loads the extracted year-aware text files (tools/sec_html_to_text.py output),
re-chunks every document with ChunkerService, and validates:

- empty chunks = 0
- duplicate chunk IDs = 0
- incorrect document_id = 0
- missing section <= 0.1%
- mid-sentence cuts <= 5%
- table-header loss <= 2% (year header preserved in multi-year table chunks)
- deterministic chunk IDs = 100% (same doc twice -> identical ids)

Writes results to docs/CHUNKING_AUDIT.json.
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.core.chunker import ChunkerService

FILES = {
    2022: "/tmp/msft_fy2022_v4.txt",
    2023: "/tmp/msft_fy2023_v4.txt",
    2024: "/tmp/msft_fy2024_v4.txt",
    2025: "/tmp/msft_fy2025_v4.txt",
    2026: "/tmp/msft_fy2026_v4.txt",
}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "CHUNKING_AUDIT.json")

_SENT_END = re.compile(r"[.!?؟]\s*$")


def chunk_doc(path: str, year: int, chunker: ChunkerService):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    doc_id = f"microsoft-10k-{year}"
    chunks = chunker.chunk_document(
        text,
        document_id=doc_id,
        source=f"10-K FY{year}",
        metadata={
            "company": "Microsoft",
            "fiscal_year": year,
            "document_type": "10-K",
            "document_id": doc_id,
        },
    )
    return chunks


def main():
    chunker = ChunkerService(chunk_size=800, chunk_overlap=100)
    results = {}
    total_chunks = 0
    total_empty = 0
    total_bad_doc_id = 0
    total_missing_section = 0
    total_mid_sentence = 0
    total_table_header_loss = 0
    total_missing_page = 0
    all_ids_run1 = []
    all_ids_run2 = []

    for year in sorted(FILES):
        path = FILES[year]
        doc_id = f"microsoft-10k-{year}"
        chunks = chunk_doc(path, year, chunker)

        empty = [c for c in chunks if not c.text or not c.text.strip()]
        bad_ids = [c for c in chunks if c.metadata.get("document_id") != doc_id]
        missing_section = [c for c in chunks if not c.metadata.get("section")]

        # mid-sentence cut: a chunk cut in the middle of a sentence. Only
        # counts when the chunk hit the size limit (real hard split) AND the
        # next chunk continues with a lowercase letter. Page-header repeats
        # ("PART I" / "Item 1A") and overlap tails are not sentence cuts.
        mid_sentence = 0
        for i, c in enumerate(chunks[:-1]):
            nxt = chunks[i + 1].text.lstrip()
            if len(c.text) >= chunker.chunk_size * 0.95 and not _SENT_END.search(c.text) and nxt and (nxt[0].islower() or nxt[0] in ",.،؛;"):
                mid_sentence += 1

        # table header loss: chunks containing a year-headed table row should
        # retain the year labels near the row labels
        table_chunks = [c for c in chunks if re.search(r"\byears?:?\s*\(In|Year Ended June 30", c.text)]
        header_lost = 0
        for c in table_chunks:
            if not re.search(r"\b(?:19|20)\d{2}\b", c.text[:400]):
                header_lost += 1

        # determinism check: re-chunk same text again, ids must match 1:1
        chunks2 = chunk_doc(path, year, chunker)
        ids1 = [c.id for c in chunks]
        ids2 = [c.id for c in chunks2]
        deterministic = ids1 == ids2

        duplicate_ids_in_doc = len(ids1) != len(set(ids1))

        stats = {
            "document_id": doc_id,
            "chunks": len(chunks),
            "empty_chunks": len(empty),
            "bad_document_id": len(bad_ids),
            "missing_section": len(missing_section),
            "missing_page": len([c for c in chunks if c.metadata.get("page") is None]),
            "mid_sentence_cuts": mid_sentence,
            "table_chunks": len(table_chunks),
            "table_header_loss": header_lost,
            "deterministic_rechunk": deterministic,
            "duplicate_ids_in_doc": duplicate_ids_in_doc,
        }
        results[year] = stats
        total_chunks += len(chunks)
        total_empty += len(empty)
        total_bad_doc_id += len(bad_ids)
        total_missing_section += len(missing_section)
        total_mid_sentence += mid_sentence
        total_table_header_loss += header_lost
        total_missing_page += stats["missing_page"]
        all_ids_run1.extend(ids1)
        all_ids_run2.extend(ids2)

    cross_run_dup = len(set(all_ids_run1)) != len(all_ids_run1) or len(set(all_ids_run2)) != len(all_ids_run2)

    summary = {
        "total_chunks": total_chunks,
        "total_empty": total_empty,
        "total_bad_document_id": total_bad_doc_id,
        "total_missing_section": total_missing_section,
        "total_missing_page": total_missing_page,
        "total_mid_sentence": total_mid_sentence,
        "total_table_header_loss": total_table_header_loss,
        "duplicate_ids_any_run": cross_run_dup,
        "deterministic_100pct": all(r["deterministic_rechunk"] for r in results.values()),
        "acceptance": {
            "empty_chunks_zero": total_empty == 0,
            "duplicate_ids_zero": not cross_run_dup,
            "bad_document_id_zero": total_bad_doc_id == 0,
            "missing_section_le_0.1pct": total_missing_section / max(total_chunks, 1) <= 0.001,
            "missing_page_le_1pct": total_missing_page / max(total_chunks, 1) <= 0.01,
            "mid_sentence_le_5pct": total_mid_sentence / max(total_chunks, 1) <= 0.05,
            "table_header_loss_le_2pct": total_table_header_loss / max(total_chunks, 1) <= 0.02,
        },
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"per_year": results, "summary": summary}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))
    for y, r in results.items():
        print(f"{y}: {r['chunks']} chunks empty={r['empty_chunks']} bad_id={r['bad_document_id']} "
              f"miss_section={r['missing_section']} mid={r['mid_sentence_cuts']} "
              f"table_loss={r['table_header_loss']} det={r['deterministic_rechunk']} dup={r['duplicate_ids_in_doc']}")


if __name__ == "__main__":
    main()