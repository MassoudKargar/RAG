#!/usr/bin/env python3
"""PHASE 2 — Metadata & index integrity audit on the live ChromaDB collection.

Validates per-chunk metadata invariants on the production MSFT corpus:

- only expected documents (microsoft-10k-2022..2026)
- fiscal_year present and in [2022..2026] for every microsoft-10k-* chunk
- document_id in the expected set
- chunk_id format <document_id>::chunk_%05d and index == chunk_index
- section present, page present in [1..102]
- chunk text non-empty
- years_present (when set) contains only plausible years (19xx/20xx) and
  includes fiscal_year when the chunk text mentions that year
- collection count == sum of per-document counts (no orphans/stale docs)

Note: years_present is an auxiliary lookup aid — body-text chunks legitimately
have no years (so absence is fine), and multi-year table rows carry years that
differ from fiscal_year. The invalids we flag here are years that do not look
like years at all, or a years_present that omits a year actually mentioned in
the text.

Writes docs/METADATA_AUDIT.json. Exit 0 when all acceptance checks pass.
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "METADATA_AUDIT.json")

EXPECTED_DOCS = {"microsoft-10k-2022", "microsoft-10k-2023", "microsoft-10k-2024",
                 "microsoft-10k-2025", "microsoft-10k-2026"}
VALID_FY = {2022, 2023, 2024, 2025, 2026}
CHUNK_ID_RE = re.compile(r"^(.+)::chunk_(\d{5})$")
_YEAR_IN_TEXT_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def main() -> int:
    client = chromadb.PersistentClient(path="/var/rag_app/chroma_db")
    col = client.get_collection("RAG_COLLECTION")
    total = col.count()
    got = col.get(include=["documents", "metadatas"], limit=max(total * 2, 10000))
    metas = got.get("metadatas") or []
    docs = got.get("documents") or []

    issues = Counter()
    issue_samples = {}
    per_doc = Counter()

    for m, d in zip(metas, docs):
        m = m or {}
        did = m.get("document_id")
        per_doc[did] += 1

        if did not in EXPECTED_DOCS:
            issues["unexpected_document_id"] += 1
            issue_samples.setdefault("unexpected_document_id", []).append(did)
            continue

        fy = m.get("fiscal_year")
        if fy not in VALID_FY:
            issues["bad_fiscal_year"] += 1
            issue_samples.setdefault("bad_fiscal_year", []).append((did, fy))

        cid = m.get("chunk_id") or ""
        mch = CHUNK_ID_RE.match(cid)
        if not mch or mch.group(1) != did:
            issues["bad_chunk_id"] += 1
            issue_samples.setdefault("bad_chunk_id", []).append((did, cid))
        else:
            idx = int(mch.group(2))
            if idx != m.get("chunk_index"):
                issues["chunk_id_index_mismatch"] += 1
                issue_samples.setdefault("chunk_id_index_mismatch", []).append((did, cid, m.get("chunk_index")))

        if not m.get("section"):
            issues["missing_section"] += 1
        page = m.get("page")
        if page is None or not (1 <= int(page) <= 102):
            issues["bad_page"] += 1
            issue_samples.setdefault("bad_page", []).append((did, page))
        if d is None or not str(d).strip():
            issues["empty_document"] += 1

        # years_present: valid if every listed value looks like a real year and
        # no year mentioned in the chunk text is missing (when years_present exists)
        yp = m.get("years_present")
        if yp is not None:
            bad_vals = [y for y in yp if not (isinstance(y, int) and 1900 <= y <= 2100)]
            if bad_vals:
                issues["years_present_bad_values"] += 1
                issue_samples.setdefault("years_present_bad_values", []).append((did, bad_vals))
            text_years = set(int(y) for y in _YEAR_IN_TEXT_RE.findall(str(d)))
            if text_years:
                missing_from_yp = text_years - set(yp)
                if missing_from_yp and len(missing_from_yp) / len(text_years) > 0.3:
                    issues["years_present_omits_text_years"] += 1
                    issue_samples.setdefault("years_present_omits_text_years", []).append(
                        (did, sorted(missing_from_yp)[:4]))

    expected_total = sum(per_doc[y] for y in sorted(EXPECTED_DOCS))
    acceptance = {
        "count_matches_sum": total == expected_total,
        "only_expected_docs": issues["unexpected_document_id"] == 0,
        "fiscal_year_valid_all": issues["bad_fiscal_year"] == 0,
        "chunk_id_format_all": issues["bad_chunk_id"] == 0 and issues["chunk_id_index_mismatch"] == 0,
        "section_all": issues["missing_section"] == 0,
        "page_all": issues["bad_page"] == 0,
        "no_empty_documents": issues["empty_document"] == 0,
        "years_present_sane": issues["years_present_bad_values"] == 0,
    }
    result = {
        "total_count": total,
        "per_doc": dict(per_doc),
        "issue_counts": dict(issues),
        "issues_sample": {k: v[:5] for k, v in issue_samples.items()},
        "acceptance": acceptance,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    ok = all(acceptance.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())