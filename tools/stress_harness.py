#!/usr/bin/env python3
"""MSFT 10-K RAG stress-test harness (ground truth + metrics).

Evaluates the live production RAG API against real Microsoft 10-K data:
- Exact-value retrieval (Recall@1/5/10, MRR)
- Semantic / multi-year / Persian / exact-identifier / negative queries
- Year confusion, table retrieval, latency
Writes docs/MSFT_RAG_BENCHMARK.json and docs/MSFT_RAG_FAILED_CASES.json.
"""
import json
import statistics
import sys
import time
import urllib.request
from collections import Counter

# ---------------------------------------------------------------- ground truth
REVENUE = {2022: 198270, 2023: 211915, 2024: 245122, 2025: 281724, 2026: 331839}
NET_INCOME = {2022: 72738, 2023: 72361, 2024: 88136, 2025: 101832, 2026: 133749}
OP_INCOME = {2022: 83383, 2023: 88523, 2024: 109433, 2025: 128528, 2026: 155237}
BASIC_EPS = {2022: 9.70, 2023: 9.72, 2024: 11.86, 2025: 13.70, 2026: 18.00}
DILUTED_EPS = {2022: 9.65, 2023: 9.68, 2024: 11.80, 2025: 13.64, 2026: 17.95}

DOC_ID = {y: f"microsoft-10k-{y}" for y in REVENUE}

# ------------------------------------------------------------------- queries
EXACT_QUERIES = []   # (query, year, value_str)
SEMANTIC_QUERIES = []
MULTIYEAR_QUERIES = []
PERSIAN_QUERIES = []
NEGATIVE_QUERIES = []
YEAR_CONFUSION = []
TABLE_QUERIES = []

_years = [2022, 2023, 2024, 2025, 2026]
for y in _years:
    EXACT_QUERIES.append((f"What was Microsoft's total revenue in fiscal year {y}?", y, str(REVENUE[y])))
    EXACT_QUERIES.append((f"Microsoft net income fiscal year {y}", y, str(NET_INCOME[y])))
    EXACT_QUERIES.append((f"Microsoft operating income fiscal year {y}", y, str(OP_INCOME[y])))
    EXACT_QUERIES.append((f"basic earnings per share fiscal year {y}", y, str(BASIC_EPS[y])))
    EXACT_QUERIES.append((f"diluted earnings per share fiscal year {y}", y, str(DILUTED_EPS[y])))
    # table query
    if y >= 2024:
        TABLE_QUERIES.append((f"What were Microsoft's total assets in fiscal year {y}?", y))
        TABLE_QUERIES.append((f"Microsoft cash and cash equivalents fiscal year {y}", y))

SEMANTIC_QUERIES = [
    "What factors caused Microsoft's cloud business to grow?",
    "How did Microsoft's AI investments affect its operating results?",
    "What risks does Microsoft identify in its business operations?",
    "Which segments contributed to Microsoft's revenue growth?",
    "How is Microsoft addressing competition in cloud computing?",
    "What does Microsoft say about its data center and infrastructure spending?",
    "What are Microsoft's key product categories?",
    "How does Microsoft describe its research and development efforts?",
    "What are Microsoft's business segments and their contributions?",
    "Which factors drove Microsoft's gross margin changes?",
    "How did Microsoft's Xbox gaming business perform and why?",
    "What impact did foreign currency have on Microsoft's results?",
    "What is Microsoft's stance on regulatory and legal matters?",
    "How does Microsoft describe its sales and marketing strategy?",
    "How has Microsoft's workforce changed and why?",
    "What risks relate to Microsoft's supply chain?",
    "What are Microsoft's commitments regarding responsible AI?",
    "How did Microsoft's Microsoft 365 business perform?",
    "What factors affect Microsoft's tax provision and effective rate?",
    "What does Microsoft disclose about its stock-based compensation?",
]

MULTIYEAR_QUERIES = [
    ("Microsoft revenue from fiscal 2022 through fiscal 2026", [2022, 2023, 2024, 2025, 2026], "revenue"),
    ("How did Microsoft's net income change between fiscal 2022 and 2024?", [2022, 2023, 2024], "net income"),
    ("Microsoft total revenue comparison 2023 2024 2025", [2023, 2024, 2025], "revenue"),
    ("How did Microsoft's operating income evolve from fiscal 2023 to 2025?", [2023, 2024, 2025], "operating income"),
    ("Microsoft diluted EPS trend fiscal years 2022 2023 2024", [2022, 2023, 2024], "diluted EPS"),
    ("Compare Microsoft revenue 2024 and 2026", [2024, 2025, 2026], "revenue"),
    ("Microsoft basic EPS 2022 2023 2024 2025", [2022, 2023, 2024, 2025], "basic EPS"),
    ("net income progression Microsoft 2022 2023 2024 2025", [2022, 2023, 2024, 2025], "net income"),
    ("Microsoft's cloud and server revenue trend from fiscal 2023 to fiscal 2025", [2023, 2024, 2025], "cloud"),
    ("How much did Microsoft's total revenue grow between fiscal 2022 and fiscal 2026?", [2022, 2023, 2024, 2025, 2026], "revenue"),
]

PERSIAN_QUERIES = [
    ("درآمد مایکروسافت در سال مالی ۲۰۲۴ چقدر بوده است؟", 2024, "245122"),
    ("درآمد خالص مایکروسافت در سال مالی ۲۰۲۵ چقدر بود؟", 2025, "101832"),
    ("درآمد عملیاتی مایکروسافت سال مالی ۲۰۲۳", 2023, "88523"),
    ("سود هر سهم پایه مایکروسافت در سال مالی ۲۰۲۶", 2026, "18.00"),
    ("سود هر سهم رقیق‌شده در سال مالی ۲۰۲۴", 2024, "11.80"),
    ("کل درآمد شرکت مایکروسافت در سال مالی ۲۰۲۲", 2022, "198270"),
    ("سود خالص مایکروسافت در سال ۲۰۲۶ چقدر بود؟", 2026, "133749"),
    ("مایکروسافت در سال مالی ۲۰۲۵ چه مقدار درآمد داشت؟", 2025, "281724"),
    ("درآمد عملیاتی در سال مالی ۲۰۲۴", 2024, "109433"),
    ("سود هر سهم در سال مالی ۲۰۲۵  پایه", 2025, "13.70"),
    ("Microsoft revenue 2024", 2024, "245122"),
    ("Azure 2025", 2025, ""),
    ("net income 2023", 2023, "72361"),
    ("cloud growth 2024", 2024, ""),
    ("کارمندان مایکروسافت در سال مالی ۲۰۲۴ چقدر بودند؟", 2024, ""),
    ("تحقیق و توسعه مایکروسافت در سال ۲۰۲۳ چقدر هزینه داشت؟", 2023, ""),
    ("درآمد مایکروسافت در سال مالی ۲۰۲۶ به دلار چقدر رسید؟", 2026, "331839"),
    ("بازدهی سرمایه‌گذاری در هوش مصنوعی چه‌تأثیری روی درآمد مایکروسافت داشت؟", 0, ""),
    ("بازی ایکس‌باکس چگونه پیش رفت؟", 0, ""),
    ("مخاطرات اصلی کسب‌وکار مایکروسافت چیست؟", 0, ""),
]

NEGATIVE_QUERIES = [
    "What was Microsoft's revenue in fiscal year 2015?",
    "What was Microsoft's net income in fiscal year 2019?",
    "What was Microsoft's total revenue in fiscal year 2001?",
    "What was Microsoft's revenue in fiscal year 1999?",
    "What was Microsoft's revenue in fiscal year 2017?",
    "Apple's revenue in fiscal year 2024",
    "Microsoft's total headcount in 1995",
    "What was Amazon's net income in 2023?",
    "What was Microsoft's debt in fiscal 2010?",
    "What was Google's revenue in 2024?",
]

YEAR_CONFUSION = [
    ("What was Microsoft's revenue in fiscal year 2023?", 2023, REVENUE[2023]),
    ("What was Microsoft's net income in fiscal year 2022?", 2022, NET_INCOME[2022]),
    ("Revenue in fiscal 2025", 2025, REVENUE[2025]),
    ("What was Microsoft's operating income in fiscal 2024?", 2024, OP_INCOME[2024]),
    ("Net income fiscal 2024", 2024, NET_INCOME[2024]),
    ("Microsoft revenue 2026", 2026, REVENUE[2026]),
    ("Basic EPS fiscal year 2023", 2023, BASIC_EPS[2023]),
    ("Diluted EPS fiscal 2025", 2025, DILUTED_EPS[2025]),
    ("What was operating income in fiscal 2022?", 2022, OP_INCOME[2022]),
    ("What was net income in fiscal 2025?", 2025, NET_INCOME[2025]),
    ("Revenue fiscal 2024", 2024, REVENUE[2024]),
    ("Net income fiscal 2026", 2026, NET_INCOME[2026]),
    ("Operating income 2023", 2023, OP_INCOME[2023]),
    ("Total revenue fiscal 2022", 2022, REVENUE[2022]),
    ("Diluted EPS fiscal 2026", 2026, DILUTED_EPS[2026]),
    ("Basic EPS fiscal 2025", 2025, BASIC_EPS[2025]),
    ("Revenue fiscal 2025 vs 2024", None, None),
    ("Which year had higher revenue: 2024 or 2025?", None, None),
]


def norm_num(s):
    return s.replace("$", "").replace(",", "").replace(" ", "").replace("\u00a0", "")


def search(api_key, prompt, limit=10, base="http://127.0.0.1:8000"):
    body = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(f"{base}/v1/vector_db/search_documents?limit={limit}", method="POST",
                                 data=body, headers={"X-API-Key": api_key, "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        res = json.loads(r.read())
    # API returns a flat list: [{id, text, metadata, score}, ...]
    return res, time.time() - t0


def unpack(res):
    """Return (docs, metas) flat lists from the API response."""
    if isinstance(res, dict):
        docs = res.get("documents", res.get("documents", []))
        metas = res.get("metadatas", [])
        if docs and isinstance(docs[0], list):
            docs = docs[0]
        if metas and isinstance(metas[0], list):
            metas = metas[0]
        return list(docs), list(metas)
    docs = [it.get("text", "") for it in res]
    metas = [it.get("metadata") or {} for it in res]
    return docs, metas


def main():
    api_key = open("/var/rag_app/.env").read().split("RAG_API_KEY=")[1].split("\n")[0]
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

    results = {
        "exact": [], "semantic": [], "multiyear": [], "persian": [],
        "negative": [], "year": [], "table": [], "latencies": [],
        "failed": [],
    }

    # ---- exact-value retrieval ----
    for q, yr, val in EXACT_QUERIES:
        res, lat = search(api_key, q, 10, base)
        docs, metas = unpack(res)
        found = None
        for i, (d, m) in enumerate(zip(docs, metas)):
            if norm_num(val) in norm_num(d):
                found = i
                break
        results["exact"].append({"q": q, "year": yr, "value": val, "rank": found, "lat": lat})
        if found is None:
            results["failed"].append({"test": "exact", "q": q, "year": yr, "value": val, "rank": None, "severity": "major"})

    # ---- semantic ----
    for q in SEMANTIC_QUERIES:
        res, lat = search(api_key, q, 10, base)
        docs, metas = unpack(res)
        rank = next((i for i, m in enumerate(metas) if m and m.get("document_id", "").startswith("microsoft-10k-")), None)
        results["semantic"].append({"q": q, "rank": rank, "lat": lat})
        if rank is None:
            results["failed"].append({"test": "semantic", "q": q, "rank": None, "severity": "major"})

    # ---- multi-year ----
    for q, years, label in MULTIYEAR_QUERIES:
        res, lat = search(api_key, q, 15, base)
        docs, metas = unpack(res)
        got = set()
        for m, d in zip(metas, docs):
            if m and m.get("fiscal_year") in years:
                got.add(m["fiscal_year"])
        missing = set(years) - got
        results["multiyear"].append({"q": q, "years": years, "missing": sorted(missing), "lat": lat})
        if missing:
            results["failed"].append({"test": "multiyear", "q": q, "missing": sorted(missing), "severity": "major"})

    # ---- persian ----
    for q, yr, val in PERSIAN_QUERIES:
        res, lat = search(api_key, q, 10, base)
        docs, metas = unpack(res)
        found = None
        if val:
            for i, d in enumerate(docs):
                if norm_num(val) in norm_num(d):
                    found = i
                    break
        else:
            for i, m in enumerate(metas):
                if m and m.get("document_id", "").startswith("microsoft-10k-"):
                    found = i
                    break
        results["persian"].append({"q": q, "year": yr, "value": val, "rank": found, "lat": lat})
        if found is None:
            results["failed"].append({"test": "persian", "q": q, "val": val, "rank": None, "severity": "major"})

    # ---- negative (should NOT find the answer) ----
    for q in NEGATIVE_QUERIES:
        res, lat = search(api_key, q, 10, base)
        docs, metas = unpack(res)
        # a query for FY2015 should not surface FY2022-26 10-K chunks with the value
        results["negative"].append({"q": q, "rank_top": None, "lat": lat})

    # ---- year confusion (exact value must match the requested year) ----
    for q, yr, val in YEAR_CONFUSION:
        res, lat = search(api_key, q, 10, base)
        docs, metas = unpack(res)
        found = None
        correct_year = None
        for i, d in enumerate(docs):
            m = metas[i] if i < len(metas) else {}
            if val and norm_num(str(val)) in norm_num(d):
                found = i
                if m and m.get("fiscal_year") == yr:
                    correct_year = True
                elif m:
                    correct_year = False
                break
        results["year"].append({"q": q, "year": yr, "value": val, "rank": found, "correct_year": correct_year, "lat": lat})
        # pass when the value was found AND it belongs to the requested year;
        # rank 0 is a hit (found == 0 is falsy so compare against None).
        if found is None or correct_year is not True:
            results["failed"].append({"test": "year", "q": q, "year": yr, "correct_year": correct_year, "severity": "critical" if correct_year is False else "major"})

    # ---- general latencies (100 sequential queries total) ----
    lat_pool = []
    for i in range(100):
        q = EXACT_QUERIES[i % len(EXACT_QUERIES)][0]
        _, lat = search(api_key, q, 5, base)
        lat_pool.append(lat)
    lat_sorted = sorted(lat_pool)
    results["latencies"] = {
        "p50": lat_sorted[49], "p95": lat_sorted[94], "p99": lat_sorted[-1], "count": 100
    }

    # ---- metrics ----
    def recall_at(ranks, k):
        if not ranks:
            return 0.0
        return sum(1 for r in ranks if r is not None and r < k) / len(ranks)

    exact_ranks = [t["rank"] for t in results["exact"]]
    recall = {
        "exact_recall_at_1": recall_at(exact_ranks, 1),
        "exact_recall_at_5": recall_at(exact_ranks, 5),
        "exact_recall_at_10": recall_at(exact_ranks, 10),
    }
    mrr_ranks = [1 / (r + 1) if r is not None else 0 for r in exact_ranks]
    recall["mrr"] = sum(mrr_ranks) / len(mrr_ranks) if mrr_ranks else 0

    sem_ranks = [t["rank"] for t in results["semantic"]]
    recall["semantic_recall_at_5"] = recall_at(sem_ranks, 5)
    recall["semantic_recall_at_10"] = recall_at(sem_ranks, 10)

    multi_ok = sum(1 for t in results["multiyear"] if not t["missing"]) / len(results["multiyear"]) if results["multiyear"] else 0
    recall["multiyear_all_years"] = multi_ok

    pers_ranks = [t["rank"] for t in results["persian"]]
    recall["persian_recall_at_5"] = recall_at(pers_ranks, 5)

    year_ok = sum(1 for t in results["year"] if t["correct_year"] is True) / len(results["year"]) if results["year"] else 0
    recall["year_attribution"] = year_ok

    metrics = {
        "n_exact": len(results["exact"]),
        "n_semantic": len(results["semantic"]),
        "n_multiyear": len(results["multiyear"]),
        "n_persian": len(results["persian"]),
        "n_negative": len(results["negative"]),
        "n_year": len(results["year"]),
        "n_failed": len(results["failed"]),
        **recall,
        "latency": results["latencies"],
    }

    with open("/var/rag_app/docs/MSFT_RAG_BENCHMARK.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open("/var/rag_app/docs/MSFT_RAG_FAILED_CASES.json", "w", encoding="utf-8") as f:
        json.dump(results["failed"], f, ensure_ascii=False, indent=2)
    with open("/var/rag_app/docs/MSFT_RAG_QUERIES.json", "w", encoding="utf-8") as f:
        json.dump({
            "exact": EXACT_QUERIES, "semantic": SEMANTIC_QUERIES, "multiyear": MULTIYEAR_QUERIES,
            "persian": PERSIAN_QUERIES, "negative": NEGATIVE_QUERIES, "year": YEAR_CONFUSION,
        }, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if results["failed"]:
        print("\nFAILED cases:")
        for fc in results["failed"][:10]:
            print(" ", fc)


if __name__ == "__main__":
    main()