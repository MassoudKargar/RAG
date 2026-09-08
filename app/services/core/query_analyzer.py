"""Query Analyzer — structured interpretation of retrieval queries.

Powers the retrieval improvements (year-aware filter, multi-year spans,
Persian vocabulary, trend/comparison intent). Design:

- ``normalize`` maps Persian/Arabic digits + Arabic characters onto Latin
  digits / standard orthography for consistent downstream matching.
- ``extract_years`` turns any year mention into a sorted list; year RANGES
  (``2022 through 2026``, ``between 2022 and 2024``, ``fiscal 2022-2026``)
  are expanded to the full span.
- ``extract_metrics`` maps financial vocabulary (English + Persian) to
  canonical metric ids: revenue, net_income, gross_margin, operating_income,
  basic_eps, diluted_eps, total_assets, cash, headcount, rnd, tax, etc.
- ``extract_company`` recognizes MSFT / Apple / Amazon / Google mentions
  (Windows/Farsi forms included) so cross-document queries can be scoped.
- ``classify`` assigns an intent: exact_value, comparison, trend, multi_year,
  semantic, negative.
"""
from typing import Dict, List, Optional, Tuple
import re

# -------------------------------------------------------------------- digits
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_ARABIC_CHARS = str.maketrans("يكأآةگپچژ", "یکاآهگپچژ")

_METRIC_MAP = {
    # revenue
    "revenue": "revenue", "total revenue": "revenue", "sales": "revenue",
    "درآمد": "revenue", "درآمد کل": "revenue", "کل درآمد": "revenue", "فروش": "revenue",
    # net income
    "net income": "net_income", "net earnings": "net_income", "profit": "net_income",
    "سود خالص": "net_income", "درآمد خالص": "net_income", "سود": "net_income",
    # operating income
    "operating income": "operating_income", "operating profit": "operating_income",
    "سود عملیاتی": "operating_income", "درآمد عملیاتی": "operating_income",
    # gross margin
    "gross margin": "gross_margin", "gross profit": "gross_margin",
    "حاشیه ناخالص": "gross_margin", "سود ناخالص": "gross_margin",
    # EPS
    "earnings per share": "eps", "per share earnings": "eps",
    "سود هر سهم": "eps", "سهام": "eps", "سهم": "eps",
    # basic EPS
    "basic eps": "basic_eps", "basic earnings per share": "basic_eps",
    "سود هر سهم پایه": "basic_eps", "eps پایه": "basic_eps", "پایه": "basic_eps",
    # diluted EPS
    "diluted eps": "diluted_eps", "diluted earnings per share": "diluted_eps",
    "سود هر سهم رقیق": "diluted_eps", "رقیق": "diluted_eps", "رقیق‌شده": "diluted_eps",
    # balance sheet
    "total assets": "total_assets", "assets": "total_assets",
    "کل دارایی": "total_assets", "دارایی": "total_assets",
    "cash and cash equivalents": "cash", "cash": "cash",
    "نقد": "cash", "نقد و معادل": "cash", "معادل نقد": "cash",
    # other
    "headcount": "headcount", "employees": "headcount", "workforce": "headcount",
    "کارمندان": "headcount", "نیروی کار": "headcount", "کارکنان": "headcount",
    "research and development": "rnd", "r&d": "rnd",
    "تحقیق و توسعه": "rnd",
    "tax": "tax", "taxes": "tax", "مخاطر مالیاتی": "tax", "مالیات": "tax",
    "interest": "interest", "بهره": "interest",
    "cloud": "cloud", "azure": "azure", "گیمینگ": "gaming", "xbox": "gaming",
    "server and tools": "server", "organization": "org_segment",
}

_COMPANY_PATTERN = re.compile(r"microsoft|msft|ماکروسافت|مایکروسافت|apple|اپل|amazon|آمازون|google|گوگل", re.IGNORECASE)
_COMPANY_MAP = {
    "microsoft": "microsoft", "msft": "microsoft", "ماکروسافت": "microsoft", "مایکروسافت": "microsoft",
    "apple": "apple", "اپل": "apple",
    "amazon": "amazon", "آمون": "amazon", "آمازون": "amazon",
    "google": "google", "گوگل": "google",
}

_QUANT_PATTERN = re.compile(r"\b(?:diluted|basic)\b", re.IGNORECASE)


class QueryAnalyzer:
    def normalize(self, text: str) -> str:
        """Normalize digits and Arabic letters for consistent matching."""
        t = text.translate(_PERSIAN_DIGITS).translate(_ARABIC_CHARS)
        return re.sub(r"\s+", " ", t).strip().lower()

    def extract_years(self, text: str) -> List[int]:
        """Return sorted unique years, expanding explicit ranges.

        Recognizes: single years (2022), ``2022 through 2026``,
        ``between 2022 and 2024``, ``fiscal 2022-2026``, ``2022 2023 2024``.
        """
        t = self.normalize(text)
        years = [int(m) for m in re.findall(r"\b(?:19|20)\d{2}\b", t)]
        if not years:
            return []

        # range spans: X through Y / X to Y / between X and Y / X-Y / X..Y.
        # "fiscal" between the years is optional ("between fiscal 2022 and
        # fiscal 2026").
        span_re = re.compile(
            r"(?:19|20)\d{2}\s*(?:fiscal\s+|fy\s+)?"
            r"(?:through|to|and|till|from|[-–—..])\s*"
            r"(?:fiscal\s+|fy\s+)?(?:19|20)\d{2}"
        )
        spans = []
        for m in span_re.finditer(t):
            a, b = [int(x) for x in re.findall(r"(?:19|20)\d{2}", m.group(0))]
            if 0 < abs(a - b) <= 15:
                spans.append((min(a, b), max(a, b)))
        # normalize single-year list to a set (order preserved)
        uni = set(years)
        for lo, hi in spans:
            uni.update(range(lo, hi + 1))
        return sorted(uni)

    def extract_metric(self, text: str) -> Optional[str]:
        """Return the canonical metric key mentioned, or None.

        EPS qualifiers (basic / diluted / پایه / رقیق) take precedence over the
        generic EPS key even when they appear detached ("سود هر سهم در سال
        مالی ۲۰۲۵ پایه").
        """
        t = self.normalize(text)
        if "سود هر سهم" in t or "earnings per share" in t or "eps" in t:
            if re.search(r"رقیق|diluted", t):
                return "diluted_eps"
            if re.search(r"پایه|basic", t):
                return "basic_eps"
            return "eps"
        # longest-prefix style: iterate over sorted map keys by length desc
        for key in sorted(_METRIC_MAP, key=len, reverse=True):
            if key in t:
                return _METRIC_MAP[key]
        return None

    def extract_company(self, text: str) -> Optional[str]:
        t = self.normalize(text)
        m = _COMPANY_PATTERN.search(t)
        if not m:
            return None
        return _COMPANY_MAP.get(m.group(0), m.group(0))

    def classify(self, text: str) -> str:
        """Intent classification.

        Returns one of: negative, start, comparison, trend, multi_year,
        single_year_exact, semantic.
        """
        t = self.normalize(text)
        years = self.extract_years(t)
        # negative: asks about company/year combos outside the corpus — the
        # analyzer can't know the corpus here; classification of 'negative' is
        # left to the caller via extract_company + corpus scoping. We detect
        # obviously off-corpus years (<=2019) as negative-candidate.
        if years and max(years) <= 2019:
            return "negative"
        if re.search(r"\b(?:change|grow|trend|evolve|progression|progress|compare|different|vs|versus|مقایسه|روند|تغییر|رشد|تکامل)\b", t):
            if len(years) >= 2:
                return "comparison" if re.search(r"\b(?:compare|versus|vs)\b", t) else "trend"
            return "semantic"
        if len(years) >= 3:
            return "multi_year"
        if len(years) == 2:
            return "multi_year"  # pair = likely comparison/span
        if len(years) == 1:
            return "single_year"
        return "semantic"

    def parse(self, text: str) -> Dict[str, object]:
        """Full structured interpretation of a query."""
        return {
            "normalized": self.normalize(text),
            "years": self.extract_years(text),
            "metric": self.extract_metric(text),
            "company": self.extract_company(text),
            "intent": self.classify(text),
        }


query_analyzer = QueryAnalyzer()