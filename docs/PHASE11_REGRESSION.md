# Phase 11 — Full Regression Benchmark

**Commit:** `ff4343f` · **Date:** 2026-09-08

## Test suite

Full pytest: **56 passed** (0 failed) in 113s.

- tests/test_chunking.py (15)
- tests/test_e2e_rag.py (3)
- tests/test_query_analyzer.py (22)
- tests/test_section_retrieval.py (7)
- tests/test_table_retrieval.py (9)

## Stress harness (103 checks)

```
n_failed: 0
exact_recall_at_1: 0.96   exact_recall_at_5: 1.0   exact_recall_at_10: 1.0
mrr: 0.9733
semantic_recall_at_5: 1.0   semantic_recall_at_10: 1.0
multiyear_all_years: 1.0
persian_recall_at_5: 1.0
year_attribution: 0.8889
latency: p50 187ms · p95 238ms · p99 803ms
```

Stable vs baseline: exact@1 0.96, MRR 0.973, multiyear 1.0, persian 1.0,
semantic 1.0 — no regressions from phases 9-10 additions.