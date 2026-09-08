# Phase 9 — Section-Aware Retrieval

**Commit:** `(pending)` · **Date:** 2026-09-08

## What was added

1. **`query_analyzer.section_for_query(text)`** — detects the SEC 10-K section a
   query asks about (EN + Persian), mapping concepts to canonical item keys:

   | Query concept | Canonical section |
   |---|---|
   | risk factors / مخاطرات / ریسک / تهدیدها / خطرات | `Item 1A` |
   | cybersecurity / امنیت سایبری / سایبر / هک | `Item 1C` |
   | Management's Discussion / MD&A / results of operations / بررسی مدیریت | `Item 7` |
   | financial statements / balance sheet / صورت‌های مالی / ترازنامه | `Item 8` |
   | business overview / کسب‌وکار / فعالیت اصلی | `Item 1` |
   | corporate governance / directors / حاکمیت شرکتی | `Item 10` |
   | controls and procedures | `Item 9A` |
   | market for registrant / common equity | `Item 5` |

   Also `RAGService._canonical_section()` maps stored chunk section labels
   (`Item 1A`, `RISK FACTORS`, `MANAGEMENT’S DISCUSSION…`) back to canonical
   items.

2. **Section boost in `sparse_weight`** — when a query has a section intent
   and a candidate chunk's canonical section matches, `w += 10`. Soft boost:
   it never hard-filters, so an exact-value query with an incidental section
   word is not hijacked (verified: "Microsoft net income fiscal year 2024"
   still ranks the values table at top).

## Tests

`tests/test_section_retrieval.py` (7 tests, live embedder, production corpus):
- section detection EN (7 subsections) + FA (4)
- no section tag for metric queries
- ranking: risk-factors → Item 1A in top 3, business overview → Item 1 in top 3,
  metric query not hijacked, combined year+metric+section query

Full suite: **47 passed**.

## Regression

Full stress harness (103 checks): **n_failed = 0**

exact@1 0.96 · MRR 0.973 · multiyear 1.0 · persian 1.0 · semantic 1.0 ·
year_attribution 0.889 — unchanged.