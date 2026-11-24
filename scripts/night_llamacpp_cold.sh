#!/usr/bin/env bash
# Faz 7p (GPU 2, ~8 GB pay, guard her adimda acik), ~3,5-4 saat: 2a'daki 7 model, prompt onbellegi kapali.
#   cd /disk3/trbench/yeni && nohup bash scripts/night_llamacpp_cold.sh > /disk3/trbench/logs/llamacpp_cold.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[7p] basladi $(date)"
bash scripts/gpu_bench.sh configs/gpu_llamacpp_cold.yaml || echo "[7p] hata"
/disk3/trbench/venv/bin/python -m trbench.eval.gpu_report --llamacpp-file gpu_llamacpp_cold.jsonl
echo "[7p] HEPSI BITTI $(date)"
