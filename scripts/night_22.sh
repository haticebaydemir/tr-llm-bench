#!/usr/bin/env bash
# 22 Kasim gecesi: (1) upstream vLLM kurulumu, (2) KV-cache dogrulamasi, (3) vLLM karsilastirmasi upstream ile.
#   cd /disk3/trbench/yeni && nohup bash scripts/night_22.sh > /disk3/trbench/logs/night_22.out 2>&1 &
# Her adim bagimsiz: biri basarisiz olursa digeri yine calisir.
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0   # Faz 7r: bir adim hata verirse rapor uretilmez, son satir HATA olur, cikis kodu 1
echo "[gece] basladi $(date)"
bash scripts/vllm_upstream_setup.sh || { echo "[gece] upstream vLLM kurulamadi"; FAIL=1; }
OUT=results/gpu_kvcache.jsonl bash scripts/gpu_quality.sh configs/gpu_kvcache2.txt || { echo "[gece] KV kosusu hata"; FAIL=1; }
if [ -x /disk3/trbench/venv-vllm-up/bin/vllm ]; then
  TRBENCH_VLLM=/disk3/trbench/venv-vllm-up/bin/vllm bash scripts/gpu_bench.sh configs/gpu_vllm_upstream.yaml \
    || { echo "[gece] upstream vLLM olcumu hata (V100'de calismiyor olabilir; log: /disk3/trbench/logs)"; FAIL=1; }
fi
if [ "$FAIL" -ne 0 ]; then
  echo "[gece] HATA: en az bir adim basarisiz, rapor uretilmedi $(date)"
  exit 1
fi
echo "[gece] HEPSI BITTI $(date)"
