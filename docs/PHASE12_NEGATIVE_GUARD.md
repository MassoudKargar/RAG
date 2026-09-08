# Phase 12 — Negative / Hallucination Guard

**Commit:** `(pending)` · **Date:** 2026-09-08

## What was added

`RAGService.search_similar_documents` now has an explicit **honest no-answer
path**:

1. **Out-of-corpus fiscal years** (via `year_filter`): "fiscal year 2015",
   "1999", "2010" etc. → empty result (no value can exist → don't fabricate).
2. **Other-company guard**: when the query names a company other than
   Microsoft (Apple, Amazon, Google...) and does **not** also mention
   Microsoft, an empty pool is returned so the LLM cannot hallucinate a
   Microsoft figure as the answer. (`query_analyzer.extract_company` +
   normalized "مایکروسافت/microsoft/ماکروسافت" check.)
   - If Microsoft is also mentioned, search proceeds ("Apple 2024 what is
     microsoft revenue" still retrieves).

## Tests

`tests/test_negative_guard.py` (10 tests):
- 4 out-of-corpus-year queries → empty
- 3 other-company queries → empty
- 3 Microsoft/English+Persian → value still retrieved

All pass.

## Regression

Full stress harness (103 checks): **n_failed = 0** — exact@1 0.96, MRR 0.973,
multiyear 1.0, persian 1.0, year_attribution 0.889. The guard does not regress
retrieval (no false negatives on Microsoft queries).