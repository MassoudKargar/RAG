"""Unit tests for the QueryAnalyzer (phase 3)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.core.query_analyzer import query_analyzer as qa


class TestYears:
    def test_single_year(self):
        assert qa.extract_years("Microsoft net income fiscal year 2024") == [2024]

    def test_persian_digits(self):
        assert qa.extract_years("درآمد مایکروسافت در سال مالی ۲۰۲۴ چقدر بوده است؟") == [2024]

    def test_through_span(self):
        assert qa.extract_years("Microsoft revenue from fiscal 2022 through fiscal 2026") == [
            2022, 2023, 2024, 2025, 2026,
        ]

    def test_between_span(self):
        assert qa.extract_years("net income change between fiscal 2022 and fiscal 2024") == [
            2022, 2023, 2024,
        ]

    def test_from_to_span(self):
        assert qa.extract_years("evolve from fiscal 2023 to 2025") == [2023, 2024, 2025]

    def test_list_years(self):
        assert qa.extract_years("Microsoft basic EPS 2022 2023 2024 2025") == [2022, 2023, 2024, 2025]


class TestMetrics:
    def test_revenue(self):
        assert qa.extract_metric("What was Microsoft's total revenue in fiscal year 2023?") == "revenue"

    def test_net_income(self):
        assert qa.extract_metric("Microsoft net income fiscal year 2022") == "net_income"

    def test_operating_income(self):
        assert qa.extract_metric("Microsoft operating income fiscal year 2024") == "operating_income"

    def test_basic_eps(self):
        assert qa.extract_metric("basic earnings per share fiscal year 2026") == "basic_eps"

    def test_diluted_eps(self):
        assert qa.extract_metric("diluted earnings per share fiscal year 2022") == "diluted_eps"

    def test_persian_revenue(self):
        assert qa.extract_metric("کل درآمد شرکت مایکروسافت در سال مالی ۲۰۲۲") == "revenue"

    def test_persian_net_income(self):
        assert qa.extract_metric("سود خالص مایکروسافت در سال ۲۰۲۶ چقدر بود؟") == "net_income"

    def test_persian_basic_eps_detached(self):
        assert qa.extract_metric("سود هر سهم در سال مالی ۲۰۲۵ پایه") == "basic_eps"

    def test_persian_diluted_eps(self):
        assert qa.extract_metric("سود هر سهم رقیق‌شده در سال مالی ۲۰۲۴") == "diluted_eps"


class TestCompanyIntent:
    def test_company_en(self):
        assert qa.extract_company("What was Microsoft's total revenue in fiscal year 2023?") == "microsoft"

    def test_company_fa(self):
        assert qa.extract_company("کل درآمد شرکت مایکروسافت در سال مالی ۲۰۲۲") == "microsoft"

    def test_intent_negative(self):
        assert qa.classify("What was Microsoft's net income in fiscal year 2019?") == "negative"

    def test_intent_comparison(self):
        assert qa.classify("Compare Microsoft revenue 2024 and 2026") == "comparison"

    def test_intent_single(self):
        assert qa.classify("Microsoft net income fiscal year 2022") == "single_year"

    def test_intent_semantic(self):
        assert qa.classify("What risks does Microsoft identify?") == "semantic"

    def test_parse_full(self):
        p = qa.parse("مایکروسافت در سال مالی ۲۰۲۵ چه مقدار درآمد داشت؟")
        assert p["years"] == [2025]
        assert p["metric"] == "revenue"
        assert p["company"] == "microsoft"