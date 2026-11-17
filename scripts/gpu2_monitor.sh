#!/usr/bin/env bash
# GPU 2'deki TUM sureclerin bellek kullanimini dakikada bir kaydeder (sadece okuma).
# Amac: GPU 2'yi paylasan servisin TEPE kullanimini olcmek.
#   nohup bash scripts/gpu2_monitor.sh > /dev/null 2>&1 &      # 24 saat calisir
#   bash scripts/gpu2_monitor.sh --summary                      # ozet
set -euo pipefail
OUT=/disk3/trbench/logs/gpu2_monitor.csv

if [ "${1:-}" = "--summary" ]; then
  /disk3/trbench/venv/bin/python - "$OUT" <<'PY'
import csv, sys, collections
rows = list(csv.reader(open(sys.argv[1])))
peak = collections.defaultdict(int)
for ts, kind, pid, name, mib in rows:
    peak[(kind, name.strip())] = max(peak[(kind, name.strip())], int(mib))
print(f"{len(rows)} satir, ilk {rows[0][0]}, son {rows[-1][0]}")
for (kind, name), mib in sorted(peak.items()):
    print(f"  {kind:6s} tepe {mib:6d} MiB  {name}")
PY
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
end=$(( $(date +%s) + 86400 ))
while [ "$(date +%s)" -lt "$end" ]; do
  ts=$(date -u +%FT%TZ)
  total=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "$ts,total,-,GPU2,$total" >> "$OUT"
  nvidia-smi -i 2 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits \
    | while IFS=, read -r pid name mib; do echo "$ts,proc,$pid,$name,${mib// /}" >> "$OUT"; done
  sleep 60
done
