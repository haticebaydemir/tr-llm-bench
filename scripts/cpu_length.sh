#!/usr/bin/env bash
# Girdi uzunlugu etkisi (CPU): her modelin Q4_K_M dosyasi icin
# okuma hizi (pp 256 / 1024 / 4096 token) ve 4096 token baglamla uretim hizi.
#   bash scripts/cpu_length.sh   -> results/cpu_length.jsonl
set -euo pipefail
trap 'echo "[length] HATA: satir $LINENO" >&2' ERR
export CUDA_VISIBLE_DEVICES=""
B=/disk3/trbench/llama.cpp/build/bin/llama-bench
LOW="nice -n 19 ionice -c 3 taskset -c 64-79"
OUT=results/cpu_length.jsonl
if pgrep -f "llama-server|eval.run|llama-bench" >/dev/null; then
  echo "[length] baska olcum calisiyor" >&2; exit 1
fi
: > "$OUT"
for F in /disk3/trbench/gguf/*/*q4_k_m*.gguf; do
  case "$F" in *-0000[2-9]-of-*) continue ;; esac   # bolunmus dosyada sadece ilk parca
  echo "[length] $(basename "$F") $(date)"
  $LOW "$B" -m "$F" -t 16 -p 256,1024,4096 -n 128 -d 0,4096 -r 3 -o jsonl >> "$OUT" 2>> /disk3/trbench/logs/llama_bench.log
done
echo "[length] bitti $(date): $(wc -l < "$OUT") satir"
