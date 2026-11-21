#!/usr/bin/env bash
# 22 Kasim gecesi: (1) upstream vLLM kurulumu, (2) KV-cache dogrulamasi, (3) vLLM karsilastirmasi upstream ile.
#   cd /disk3/trbench/yeni && nohup bash scripts/night_22.sh > /disk3/trbench/logs/night_22.out 2>&1 &
# Her adim bagimsiz: biri basarisiz olursa digeri yine calisir.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[gece] basladi $(date)"
bash scripts/vllm_upstream_setup.sh || echo "[gece] upstream vLLM kurulamadi"
OUT=results/gpu_kvcache.jsonl bash scripts/gpu_quality.sh configs/gpu_kvcache2.txt || echo "[gece] KV kosusu hata"
if [ -x /disk3/trbench/venv-vllm-up/bin/vllm ]; then
  TRBENCH_VLLM=/disk3/trbench/venv-vllm-up/bin/vllm bash scripts/gpu_bench.sh configs/gpu_vllm_upstream.yaml \
    || echo "[gece] upstream vLLM olcumu hata (V100'de calismiyor olabilir; log: /disk3/trbench/logs)"
fi
echo "[gece] HEPSI BITTI $(date)"
