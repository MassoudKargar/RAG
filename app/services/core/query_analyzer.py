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

    def extract_section(self, text: str) -> Optional[str]:
        """Return the canonical section key (e.g. 'Item 1A') a query asks about."""
        return section_for_query(text)

    def parse(self, text: str) -> Dict[str, object]:
        """Full structured interpretation of a query."""
        return {
            "normalized": self.normalize(text),
            "years": self.extract_years(text),
            "metric": self.extract_metric(text),
            "company": self.extract_company(text),
            "section": self.extract_section(text),
            "intent": self.classify(text),
        }


# Table-row labels per canonical metric. When a query names a financial
# metric, a chunk that contains the corresponding table row label ("Revenue:",
# "Net income:", "Diluted:") carries the most exact evidence — used by the
# retrieval re-ranker as a strong lexical boost.
# Order matters: longer/more specific labels first.
METRIC_LABEL_PATTERNS: Dict[str, List[str]] = {
    "revenue": [
        r"(?<!Total\s)Revenue\s*:",
        r"Total\s+revenue\s*:",
        r"Product\s+and\s+service\s+revenue\s*:",
    ],
    "net_income": [
        r"Net\s+income\s*:",
        r"Net\s+earnings\s*:",
    ],
    "operating_income": [
        r"Operating\s+income\s*:",
    ],
    "gross_margin": [
        r"Gross\s+margin\s*:",
        r"Gross\s+profit\s*:",
    ],
    "basic_eps": [
        r"Basic\s*:",
        r"Basic\s+earnings\s+per\s+share",
    ],
    "diluted_eps": [
        r"Diluted\s*:",
        r"Diluted\s+earnings\s+per\s+share",
    ],
    "total_assets": [
        r"Total\s+assets\s*:",
    ],
    "cash": [
        r"Cash\s+and\s+cash\s+equivalents\s*:",
    ],
}


def metric_label_regexes(metric: Optional[str]) -> List[str]:
    """Return the table-row label regexes for a canonical metric."""
    if not metric:
        return []
    return METRIC_LABEL_PATTERNS.get(metric, [])


query_analyzer = QueryAnalyzer()

# -------------------------------------------------------------------- sections
# SEC 10-K section mapping. When a query names a concept, map it to the
# canonical section(s) so retrieval can boost chunks from that section.
# Keys: canonical section names (as stored in chunk metadata, or the abstract
# name we map onto item numbers). Values: regexes (EN + FA).
_SECTION_ALIASES = {
    "Item 1A": [
        r"risk\s*factor", r"risks?", r"مخاطرات?|مخاطره|ریسک|تهدیدها|خطرات",
    ],
    "Item 1C": [
        r"cyber\w*", r"سایبر", r"امنیت\s*سایبری", r"هک",
    ],
    "Item 7": [
        r"management['’]?s discussion", r"\bmda\b", r"results of operations",
        r"financial\s+condition", r"operating\s+results",
        r"بررسی مدیریت|بحث مدیریت",
    ],
    "Item 8": [
        r"financial\s+statements", r"income\s+statement", r"balance\s+sheet",
        r"statements?\s+and\s+supplementary", r"صورت.*مالی|اظهارنامه مالی|ترازنامه",
    ],
    "Item 9A": [
        r"controls and procedures", r"کوادکینگ|شویه\s*های کنترل",
    ],
    "Item 1": [
        r"business\s+overview", r"the\s+business", r"business\s*:",
        r"what\s+does\s+microsoft\s+do", r"company\s+overview",
        r"کسبوکار(?!\s*اصلی)|کسب\s*و\s*کار((?!اصلی).)*$|فعالیت\s*اصلی|شرح\s*کسبوکار|کسبوکار شرکت",
    ],
    "Item 5": [
        r"market for registrant", r"common equity", r"stockholder matters",
        r"خریداران|بازار\s*صکوک*|بازار سهم",
    ],
    "Item 10": [
        r"corporate governance", r"directors", r"executive officers",
        r"حاکمیت شرکتی|مدیران",
    ],
}

# Section-intent check order for ambiguous Persian/English concepts: specific
# sections (Risk, Cybersecurity) win when their token co-occurs with a generic
# one (e.g. "مخاطرات اصلی کسبوکار" = Item 1A not Item 1).
_SECTION_CHECK_ORDER = ["Item 1C", "Item 1A", "Item 7", "Item 8", "Item 9A", "Item 1", "Item 5", "Item 10"]

_SECTION_ITEM_NAMES = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 1B": "Unresolved Staff Comments",
    "Item 1C": "Cybersecurity",
    "Item 7": "Management's Discussion and Analysis",
    "Item 7A": "Market Risk Disclosures",
    "Item 8": "Financial Statements",
    "Item 9A": "Controls and Procedures",
}


def section_for_query(text: str) -> Optional[str]:
    """Return the canonical section key (e.g. 'Item 1A') a query asks about,
    or None. Matches EN + FA concept aliases."""
    t = query_analyzer.normalize(text)
    # direct item mention: "Item 1A", "item 8"
    m = re.search(r"item\s*(\d+[a-z]?)", t)
    if m:
        return f"Item {m.group(1).upper()}"
    for section in _SECTION_CHECK_ORDER:
        for p in _SECTION_ALIASES[section]:
            if re.search(p, t):
                return section
    return None