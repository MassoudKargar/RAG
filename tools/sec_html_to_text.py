#!/usr/bin/env python3
"""Extract clean text from SEC HTML filings for RAG ingestion.

Converts HTML (e.g. Inline XBRL 10-K) into plain text that chunks well:
- removes <script>/<style>
- renders each <table> as one line per row with **year-bound values**:
  when a header row of fiscal years (e.g. "2026  2025  2024") is detected, each
  data row becomes "label: 2026: v1  2025: v2  2024: v3" so numbers stay
  attached to their column year. Without this, "Net income: $ 72,738" loses its
  year and exact-value retrieval fails.
- collapses whitespace and excess blank lines

Usage:
    python tools/sec_html_to_text.py <input.html> <output.txt>
"""
import re
import sys

from bs4 import BeautifulSoup

_YEAR_RE = re.compile(r"^\s*(?:19|20)\d{2}\s*$")
_CURRENCY_RE = re.compile(r"^[\$€£¥₹,;&#x2007\s]*$|^\$")
# trailing label columns that accompany a year header row (e.g. "Percentage
# Change", "Change", "%"): not years, not data values — ignored for alignment
_LABEL_COLS = ("percentage change", "change", "growth", "%")

# Inline XBRL metadata noise lines that pollute extracted text: taxonomy
# member references (us-gaap:...Member), CIK/registrant numbers, ISO dates,
# XBRL namespace URIs, and pure namespace-prefixed tokens. These never carry
# financial content and break sentence flow when chunked.
_MEMBER_NOISE_RE = re.compile(
    r"^(?:"
    r"(?:[\w-]+:){1,3}[\w.]*(?:Member|Fact|Axis|Domain|LineItems|Table|Abstract)?$"
    r"|\d{4}-\d{2}-\d{2}"
    r"|0{3,}\d{5,}"
    r"|https?://\S+"
    r"|\d{2}:\d{2}"
    r")\s*$"
)

_AXIS_PREFIX_RE = re.compile(r"^(?:us-gaap|msft|srt|dei|sci|country|utr|dtr|ecd|ix|link|xbrli|xbrldi|xbrldt):")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _cell_text(cell) -> str:
    return cell.get_text(" ", strip=True)


def _is_label_column(c: str) -> bool:
    return c.lower() in _LABEL_COLS or c.strip().endswith("%")


def _looks_like_year_header(cells):
    """Heuristic: a header row where most non-empty cells are bare years."""
    non_empty = [c for c in cells if c]
    if not non_empty:
        return False
    years = [c for c in non_empty if _YEAR_RE.match(c)]
    # ignore trailing label columns (Percentage Change / %)
    meaningful = [c for c in non_empty if not _is_label_column(c)]
    return len(years) >= 2 and len(years) >= 0.5 * len(meaningful)


def _tables_to_rows(soup: "BeautifulSoup") -> None:
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [_cell_text(c) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if not cells:
                continue
            rows.append(cells)

        # detect a year header row (may be the first or second row)
        year_header_idx = None
        year_header = None
        for idx, cells in enumerate(rows[:3]):
            if _looks_like_year_header(cells):
                year_header_idx = idx
                year_header = cells
                break

        lines_out = []
        for idx, cells in enumerate(rows):
            if idx == year_header_idx:
                # keep the header itself (useful for context)
                lines_out.append("years: " + "  ".join(cells))
                continue
            label = cells[0]
            rest = cells[1:]
            if year_header is not None:
                years = [c for c in year_header if _YEAR_RE.match(c)]
                if rest and len(years) >= 2:
                    # drop bare-currency cells ("$") — they only annotate the
                    # following number and would misalign year->value mapping
                    rest = [c for c in rest if not _CURRENCY_RE.match(c.strip())]
                    # drop trailing percentage/% label columns beyond the years
                    while len(rest) > len(years) and _is_label_column(str(rest[-1])):
                        rest = rest[:-1]
                    if len(rest) > len(years):
                        rest = rest[: len(years)]
                    vals = []
                    for j, v in enumerate(rest):
                        y = years[j] if j < len(years) else years[-1]
                        vals.append(f"{y}: {v}")
                    lines_out.append(f"{label}: " + "  ".join(vals))
                    continue
            values = "  ".join(rest)
            lines_out.append(f"{label}: {values}" if values else label)
        if lines_out:
            table.replace_with("\n".join(lines_out))


def _is_noise_line(s: str) -> bool:
    """True when a stripped line is pure Inline XBRL metadata noise."""
    if not s:
        return False
    if _MEMBER_NOISE_RE.match(s):
        return True
    # member/axis tokens like "us-gaap:PerformanceSharesMember" or
    # "msft:AmyEHoodMember"
    if _AXIS_PREFIX_RE.match(s) and len(s) <= 120 and ":" in s:
        return True
    if _ISO_DATE_RE.match(s):
        return True
    return False


def extract(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    _tables_to_rows(soup)
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    # drop Inline XBRL metadata lines (never financial content)
    text = "\n".join(
        ln for ln in text.split("\n") if not _is_noise_line(ln.strip())
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    out = extract(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {len(out)} chars -> {sys.argv[2]}")