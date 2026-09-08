# Phase 13 — Cross-Document Retrieval

**Commit:** `(pending)` · **Date:** 2026-09-08

## What

Verified that queries whose answer requires values from **multiple** Microsoft
10-K filings (different fiscal years) retrieve every needed year's table value
in a single shot.

The multi-year `year_filter` (`{"fiscal_year": {"$in": [...]}}`) plus the
metric-label boost already carry each comparison/trend query to the right
chunks across documents.

## Tests

`tests/test_cross_document.py` (4 tests): each query spans 2-3 fiscal years and
asserts every requested year's revenue/net-income value is present in the top-12
pool. All pass.

Examples verified:
- "revenue compare FY2022 vs FY2026" → both values retrievable
- "net income 2023 vs 2025" → both
- "revenue trend 2022 → 2024" → all three years
- "revenue 2024 vs 2026 + net income 2024" → covered

## Regression

Harness still n_failed = 0 (exact@1 0.96, multiyear 1.0, persian 1.0).