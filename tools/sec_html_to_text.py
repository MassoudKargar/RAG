#!/usr/bin/env python3
"""Extract clean text from SEC HTML filings for RAG ingestion.

Converts HTML (e.g. Inline XBRL 10-K) into plain text that chunks well:
- removes <script>/<style>
- renders each <table> as one line per row:  "label: v1  v2  v3"
  (row-preserving keeps financial figures next to their labels; cell-wise
   extraction would tear numbers away from their captions and break search)
- collapses whitespace and excess blank lines

Usage:
    python tools/sec_html_to_text.py <input.html> <output.txt>
"""
import re
import sys

from bs4 import BeautifulSoup


def _tables_to_rows(soup: "BeautifulSoup") -> None:
    for table in soup.find_all("table"):
        rows_out = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if not cells:
                continue
            label = cells[0]
            values = "  ".join(cells[1:])
            rows_out.append(f"{label}: {values}" if values else label)
        if rows_out:
            table.replace_with("\n".join(rows_out))


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