#!/bin/bash
# PHASE 1 background runner: re-extract 10-K files with XBRL noise filter,
# re-run chunking audit, update state file.
set -e
cd /var/rag_app
export HOME=/root

echo "[phase1] re-extracting 5 filings with XBRL-filtered extractor"
for fy in 2022 2023 2024 2025 2026; do
  .venv/bin/python tools/sec_html_to_text.py /tmp/msft_fy${fy}.html /tmp/msft_fy${fy}_v4.txt
done

echo "[phase1] chunking audit"
.venv/bin/python tools/chunking_audit.py

echo "[phase1] update state file"
.venv/bin/python - <<'EOF'
import json
p = "/var/rag_app/docs/BENCHMARK_STATE.json"
d = json.load(open(p))
d["phase_status"] = "audit_ready"
d["resume_command"] = "cd /var/rag_app && git add tools/ app/ && git commit -m 'phase1: XBRL noise filter + section/page chunk metadata' && systemctl restart rag-api"
json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
print("state updated")
EOF

echo "[phase1] PHASE1 SCRIPT COMPLETE"