#!/usr/bin/env bash
# Faz 7p (GPU 2, ~8 GB pay, guard her adimda acik), ~3,5-4 saat: 2a'daki 7 model, prompt onbellegi kapali.
#   cd /disk3/trbench/yeni && nohup bash scripts/night_llamacpp_cold.sh > /disk3/trbench/logs/llamacpp_cold.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0   # Faz 7r: bir adim hata verirse rapor uretilmez, son satir HATA olur, cikis kodu 1
PY="${TRBENCH_PY:-/disk3/trbench/venv/bin/python}"   # testte sahte python ile degistirilir
echo "[7p] basladi $(date)"
bash scripts/gpu_bench.sh configs/gpu_llamacpp_cold.yaml || { echo "[7p] hata"; FAIL=1; }
if [ "$FAIL" -ne 0 ]; then
  echo "[7p] HATA: en az bir adim basarisiz, rapor uretilmedi $(date)"
  exit 1
fi
"$PY" -m trbench.eval.gpu_report --llamacpp-file gpu_llamacpp_cold.jsonl || { echo "[7p] rapor hata"; FAIL=1; }
if [ "$FAIL" -ne 0 ]; then
  echo "[7p] HATA: rapor uretimi basarisiz $(date)"
  exit 1
fi
echo "[7p] HEPSI BITTI $(date)"
