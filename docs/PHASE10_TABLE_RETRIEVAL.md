# Phase 10 — Table Retrieval (Balance Sheet / Cash Flow)

**Commit:** `(pending)` · **Date:** 2026-09-08

## What was added

Extended `query_analyzer` financial vocabulary + table-row label regexes so
balance-sheet and cash-flow concepts get the exact-label boost (same
mechanism as phase 5, applied to the full statements):

| Query concept | Canonical metric | Label regex (exact evidence) |
|---|---|---|
| total assets / دارایی | `total_assets` | `Total assets:` |
| cash and cash equivalents / نقد | `cash` | `Cash and cash equivalents:` |
| stockholders' equity / حقوق صاحبان سهام | `stockholders_equity` | `Total stockholders' equity:` |
| total liabilities / بدهی‌ها | `total_liabilities` | `Total liabilities:` |
| long-term debt / بدهی بلندمدت | `long_term_debt` | `Long-term debt:` |
| goodwill / سرقفلی | `goodwill` | `Goodwill:` |
| cash from operations / جریان نقدی عملیاتی | `cash_flow_operating` | `Net cash from operations:` |
| cash from investing | `cash_flow_investing` | `Net cash from investing:` |
| cash from financing | `cash_flow_financing` | `Net cash from financing:` |

## Tests

`tests/test_table_retrieval.py` (9 tests) — 8 ground-truth queries against the
real corpus + metric-label mapping check. **9/9 passed.**

Verified ranks: total assets FY24 rank 0 · stockholders equity FY24 rank 0 ·
long-term debt FY24 rank 0 · goodwill FY26 rank 0 · total liabilities rank 0 ·
cash rank 1 · net cash from operations rank 1 · total assets FY26 rank 0.

(Note: total assets FY2026 = **758,376** verified in the corpus; earlier
745,927 in scratch ground truth was wrong.)

## Regression

Full stress harness (103 checks): **n_failed = 0** — exact@1 0.96, MRR 0.973,
multiyear 1.0, persian 1.0, semantic 1.0, year_attribution 0.889.