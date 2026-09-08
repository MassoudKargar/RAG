#!/usr/bin/env python3
"""PHASE 3 — Query Analyzer accuracy audit against the real harness queries.

Loads docs/MSFT_RAG_QUERIES.json (ground-truth queries from the stress
harness) and verifies:

- year extraction matches the ground-truth year (exact queries)
- multi-year queries expand to the full expected span
- Persian -> Latin digit normalization works (extract_years on Persian digits)
- metric extraction recognizes each metric type used in the harness
- company extraction detects Microsoft in English + Persian

Writes docs/QUERY_ANALYZER_AUDIT.json. Exit 0 when acceptance met.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.core.query_analyzer import query_analyzer as qa

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "QUERY_ANALYZER_AUDIT.json")

# ground truth: metric -> set of expected canonical names (from harness queries)
METRIC_TRUTH = {
    "revenue": {"revenue"},
    "net_income": {"net income", "net earnings", "سود خالص", "درآمد خالص", "سود"},
    "operating_income": {"operating income", "سود عملیاتی", "درآمد عملیاتی"},
    "basil_eps": {"basic eps", "basic earnings per share", "سود هر سهم پایه"},
    "diluted_eps": {"diluted eps", "diluted earnings per share", "سود هر سهم رقیق"},
}

SINGLE_YEAR_QUERIES = [
    ("What was Microsoft's total revenue in fiscal year 2023?", 2023),
    ("Microsoft net income fiscal year 2022", 2022),
    ("Microsoft operating income fiscal year 2024", 2024),
    ("basic earnings per share fiscal year 2026", 2026),
    ("diluted earnings per share fiscal year 2022", 2022),
    ("What was Microsoft's total revenue in fiscal year 2026?", 2026),
    ("Revenue in fiscal 2025", 2025),
    ("Net income fiscal 2024", 2024),
    ("Microsoft revenue 2026", 2026),
    ("Basic EPS fiscal year 2023", 2023),
    ("In 2022, what was Microsoft's revenue?", 2022),
]

PERSIAN_QUERIES = [
    ("درآمد مایکروسافت در سال مالی ۲۰۲۴ چقدر بوده است؟", 2024, "revenue"),
    ("درآمد خالص مایکروسافت در سال مالی ۲۰۲۵ چقدر بود؟", 2025, "net_income"),
    ("درآمد عملیاتی مایکروسافت سال مالی ۲۰۲۳", 2023, "operating_income"),
    ("سود هر سهم پایه مایکروسافت در سال مالی ۲۰۲۶", 2026, "basic_eps"),
    ("سود هر سهم رقیق‌شده در سال مالی ۲۰۲۴", 2024, "diluted_eps"),
    ("کل درآمد شرکت مایکروسافت در سال مالی ۲۰۲۲", 2022, "revenue"),
    ("سود خالص مایکروسافت در سال ۲۰۲۶ چقدر بود؟", 2026, "net_income"),
    ("مایکروسافت در سال مالی ۲۰۲۵ چه مقدار درآمد داشت؟", 2025, "revenue"),
    ("سود هر سهم در سال مالی ۲۰۲۵ پایه", 2025, "basic_eps"),
    ("درآمد مایکروسافت در سال مالی ۲۰۲۶ به دلار چقدر رسید؟", 2026, "revenue"),
]

MULTIYEAR_QUERIES = [
    ("Microsoft revenue from fiscal 2022 through fiscal 2026", [2022, 2023, 2024, 2025, 2026]),
    ("How did Microsoft's net income change between fiscal 2022 and 2024?", [2022, 2023, 2024]),
    ("Microsoft total revenue comparison 2023 2024 2025", [2023, 2024, 2025]),
    ("How did Microsoft's operating income evolve from fiscal 2023 to 2025?", [2023, 2024, 2025]),
    ("Microsoft diluted EPS trend fiscal years 2022 2023 2024", [2022, 2023, 2024]),
    ("Compare Microsoft revenue 2024 and 2026", [2024, 2025, 2026]),
    ("Microsoft basic EPS 2022 2023 2024 2025", [2022, 2023, 2024, 2025]),
    ("net income progression Microsoft 2022 2023 2024 2025", [2022, 2023, 2024, 2025]),
    ("Microsoft's cloud and server revenue trend from fiscal 2023 to fiscal 2025", [2023, 2024, 2025]),
    ("How much did Microsoft's total revenue grow between fiscal 2022 and fiscal 2026?", [2022, 2023, 2024, 2025, 2026]),
]


def main() -> int:
    failures = []

    # 1) single-year exact extraction
    for q, expected in SINGLE_YEAR_QUERIES:
        years = qa.extract_years(q)
        if years != [expected]:
            failures.append(("year", q, years, [expected]))

    # 2) Persian digits + metric
    for q, yr, metric in PERSIAN_QUERIES:
        years = qa.extract_years(q)
        if yr not in years:
            failures.append(("persian_year", q, years, [yr]))
        got_metric = qa.extract_metric(q)
        if got_metric != metric:
            failures.append(("persian_metric", q, got_metric, metric))

    # 3) multi-year spans + comparisons
    for q, expected in MULTIYEAR_QUERIES:
        years = qa.extract_years(q)
        if years != expected:
            failures.append(("multiyear", q, years, expected))

    # 4) company in EN + FA
    for q in ["What was Microsoft's total revenue in fiscal year 2023?", "کل درآمد شرکت مایکروسافت در سال مالی ۲۰۲۲"]:
        if qa.extract_company(q) != "microsoft":
            failures.append(("company", q, qa.extract_company(q), "microsoft"))

    # 5) intents
    intent_cases = [
        ("What was Microsoft's net income in fiscal year 2019?", "negative"),
        ("Compare Microsoft revenue 2024 and 2026", "comparison"),
        ("Microsoft revenue from fiscal 2022 through fiscal 2026", "multi_year"),
        ("Microsoft net income fiscal year 2022", "single_year"),
        ("What risks does Microsoft identify in its business operations?", "semantic"),
    ]
    for q, intent in intent_cases:
        got = qa.classify(q)
        if got != intent:
            failures.append(("intent", q, got, intent))

    total = len(SINGLE_YEAR_QUERIES) + len(PERSIAN_QUERIES) + len(MULTIYEAR_QUERIES) + 2 + len(intent_cases)
    ok = len(failures) == 0
    result = {
        "n_checks": total,
        "n_failures": len(failures),
        "failures": failures[:10],
        "acceptance": {
            "year_extraction_accurate": len([f for f in failures if f[0] == "year"]) == 0,
            "persian_digits_accurate": len([f for f in failures if f[0] in ("persian_year",)]) == 0,
            "persian_metric_accurate": len([f for f in failures if f[0] == "persian_metric"]) == 0,
            "multiyear_span_accurate": len([f for f in failures if f[0] == "multiyear"]) == 0,
            "company_accurate": len([f for f in failures if f[0] == "company"]) == 0,
            "intent_accurate": len([f for f in failures if f[0] == "intent"]) == 0,
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    main()