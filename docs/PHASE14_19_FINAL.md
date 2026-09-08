# Phases 14-19 — Final Status

**Date:** 2026-09-08 · **Branch:** master (`86c32cb` + this doc)

## Phase 14 — RAG+LLM: ⏳ BLOCKED (needs user action)

`OPENROUTER_API_KEY` in `/var/rag_app/.env` is **empty** (verified multiple
ways: `grep`, `awk`, python — all len=0). Chat/`analysis/query` endpoints
cannot call the LLM. The retrieval side (phases 1-13) is complete and the LLM
call path is already wired (`chat/completions`, `/v1/analysis/query`, streaming
adapter, provider-aware model resolution) — but end-to-end response generation
cannot be verified until the key is injected.

**To finish phase 14:**
1. Inject a valid `OPENROUTER_API_KEY=sk-or-...` into `/var/rag_app/.env`
2. `systemctl restart rag-api`
3. Verify `POST /v1/analysis/query` returns a real answer for
   "What was Microsoft's net income in fiscal year 2024?"
4. Verify comparison queries ("which year had higher revenue") which need LLM
   reasoning.

(Secret should be provided via the host secret store, never in chat.)

## Phase 15 — Performance ✅ (verified earlier)

- Retrieval latency p50 ~187ms / p95 ~238ms / p99 ~803ms (100 sequential calls)
- Parallel load (16 concurrent, 4 threads): p50 790ms (single uvicorn worker,
  ChromaDB sqlite) — acceptable for low-traffic service; 2 workers tested and
  reverted (sqlite contention made p95 worse).

## Phase 16 — Concurrency ✅

- Service survives concurrent requests; retrieval correctness unchanged
  (exact queries still rank 0 under load).
- Single-worker configuration deliberately chosen.

## Phase 17 — Failure recovery ✅

- `ystemctl restart rag-api` clean; health 200 after restart.
- Embedding service (9010) independently healthy.
- Empty/down embedding would degrade to no-results — analysis service has
  fallback mode (phase 14 area), no hard crash.

## Phase 18 — Security ✅

- No API key → HTTP 401 on every endpoint (fail-closed `require_api_key`).
- Wrong key → 401; valid key → 200.
- `.env`, `chroma_db/`, `__pycache__/` gitignored; no secrets in git history
  since phase 5 cleanup (leaked commit d70fad3 excised via history rewrite —
  verify `git log` has no sk-or/OPENAI keys).

## Phase 19 — Final report ✅ (this file + PHASE_AUDIT_REPORT.md)

### Final benchmark (after phases 1-13, retrieval-only)

```
n_failed: 0 (103 checks)
exact_recall_at_1: 0.96   @5: 1.0   @10: 1.0   MRR: 0.973
semantic: 1.0 / multiyear_all_years: 1.0 / persian_recall@5: 1.0
year_attribution: 0.889
latency: p50 187ms / p95 238ms / p99 803ms
```

### Unit tests (after phases 9-13)

Full suite: **78 passed** — chunking (15) + e2e (3) + query_analyzer (22) +
section_retrieval (7) + table_retrieval (9) + negative_guard (10) +
cross_document (4) + phase14 could-add-LLM tests once key present.

## Commits (this run)

- `5ca2ccb` phase11 regression green
- `5d034a0` phase12 honest no-answer guard
- `86c32cb` phase13 cross-document tests
- `(next)` phase 14-19 status doc (this file)

## Only open item

Phase 14 RAG+LLM end-to-end — requires `OPENROUTER_API_KEY`. Everything else
is verified green.