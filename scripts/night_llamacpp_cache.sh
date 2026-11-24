#!/usr/bin/env bash
# Faz 7m (GPU 2, ~8 GB pay, guard her adimda acik), ~40 dk: llama.cpp TTFT, prompt onbellegi acik ve kapali.
#   cd /disk3/trbench/yeni && nohup bash scripts/night_llamacpp_cache.sh > /disk3/trbench/logs/llamacpp_cache.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0   # Faz 7r: bir adim hata verirse rapor uretilmez, son satir HATA olur, cikis kodu 1
echo "[7m] basladi $(date)"
for v in nocache cache; do
  bash scripts/gpu_bench.sh "configs/gpu_llamacpp_$v.yaml" || { echo "[7m] $v hata"; FAIL=1; }
done
if [ "$FAIL" -ne 0 ]; then
  echo "[7m] HATA: en az bir adim basarisiz, rapor uretilmedi $(date)"
  exit 1
fi
/disk3/trbench/venv/bin/python -m trbench.eval.queue_report --file results/gpu_llamacpp_cache.jsonl --by-repeat
echo "[7m] HEPSI BITTI $(date)"
