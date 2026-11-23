#!/usr/bin/env bash
# Faz 7a gecesi (GPU 2, ~8 GB pay, guard her adimda acik), ~2 saat. Prefix cache kapali: 2048 -> 4096 -> 1024;
# kontroller (2048): prefix cache acik (7k, tekrar 0) ve /metrics sorgusu kapali.
#   cd /disk3/trbench/yeni && nohup bash scripts/night_mnbt.sh > /disk3/trbench/logs/night_mnbt.out 2>&1 &
# Her adim bagimsiz: biri basarisiz olursa digeri calisir. Servis etkilenirse guard o adimi durdurur.
set -uo pipefail
cd "$(dirname "$0")/.."
export TRBENCH_VLLM=/disk3/trbench/venv-vllm-up/bin/vllm
echo "[7a] basladi $(date)"
for n in 2048 2048_cache 2048_nometrics 4096 1024; do
  bash scripts/gpu_bench.sh "configs/gpu_6d_mnbt$n.yaml" || echo "[7a] mnbt$n hata"
done
/disk3/trbench/venv/bin/python -m trbench.eval.queue_report --file results/gpu_6d_mnbt.jsonl
/disk3/trbench/venv/bin/python -m trbench.eval.queue_report --file results/gpu_6d_mnbt.jsonl --by-repeat
echo "[7a] HEPSI BITTI $(date)"
