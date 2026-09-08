# MSFT RAG Baseline (Pre-Retrieval-Upgrade)

**Commit:** `b7571cf` · **Date:** 2026-09-08 · **Corpus:** Microsoft 10-K FY2022–FY2026 (3,387 chunks)

## Environment

| Setting | Value |
|---|---|
| Provider | openrouter |
| Embedding provider | local (`intfloat/multilingual-e5-small`, 384-dim) |
| Chat model | `deepseek/deepseek-v4-flash-0731` |
| Retrieval candidates | 800 |
| Retrieval K | 10 |

## Retrieval Metrics

| Metric | Baseline |
|---|---|
| Exact Recall@1 | **68%** |
| Exact Recall@5 | **92%** |
| Exact Recall@10 | **100%** |
| MRR | **0.806** |
| Semantic Recall@5 | 100% |
| Semantic Recall@10 | 100% |
| Multi-year all-years | **20%** |
| Persian Recall@5 | **70%** |
| Year attribution | **72.2%** |

## Retrieval Latency (ms)

| P50 | P95 | P99 | Count |
|---|---|---|---|
| 147 | 177 | 556 | 100 |

## Failed Cases (24 total)

### Multi-year (8 failing) — major
- "Microsoft revenue from fiscal 2022 through fiscal 2026" — missing 2023/2024/2025
- "How did Microsoft's net income change between fiscal 2022 and 2024?" — missing 2023
- "How did Microsoft's operating income evolve from fiscal 2023 to 2025?" — missing 2024
- "Compare Microsoft revenue 2024 and 2026" — missing 2025
- "Microsoft basic EPS 2022 2023 2024 2025" — missing 2022
- "net income progression Microsoft 2022 2023 2024 2025" — missing 2022/2023
- "Microsoft's cloud and server revenue trend from fiscal 2023 to fiscal 2025" — missing 2024
- "How much did Microsoft's total revenue grow between fiscal 2022 and fiscal 2026?" — missing 2023/2024/2025

### Persian (2 failing)
- "سود هر سهم پایه مایکروسافت در سال مالی ۲۰۲۶" (basic EPS FY2026 = 18.00) — not found
- "سود خالص مایکروسافت در سال ۲۰۲۶ چقدر بود؟" (net income FY2026 = 133749) — not found

### Other categories
- 14 remaining failures in exact/semantic/persian bands (delta between recall numbers above)

## Key Weaknesses Identified
1. **Multi-year queries**: single-shot retrieval with a year `$in` filter does not guarantee
   coverage of every requested year in the top-N.
2. **Persian exact queries**: no Persian → English financial vocabulary normalization;
   "سود خالص" etc. do not map to "net income" signals.
3. **Year attribution**: wrong-year chunks sometimes rank at the top for year-specific queries
   (only 72% attribution).

Baseline is the reference point; every subsequent phase must not regress these metricস by more than 5%.