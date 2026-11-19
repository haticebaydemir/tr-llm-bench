#!/usr/bin/env bash
# CPU hiz olcumu (Pareto grafigi icin): her GGUF icin llama-bench.
# Kalite olcumuyle ayni kosullar: 16 cekirdek (64-79), en dusuk oncelik, GPU yok.
#   bash scripts/cpu_speed.sh
# Cikti: results/cpu_speed.jsonl (model dosyasi basina pp512 ve tg128, 3 tekrar)
set -euo pipefail
trap 'echo "[speed] HATA: satir $LINENO, komut: $BASH_COMMAND" >&2' ERR
[ -f /disk3/trbench/env.local ] && source /disk3/trbench/env.local
export CUDA_VISIBLE_DEVICES=""
LCPP=/disk3/trbench/llama.cpp
LOW="nice -n 19 ionice -c 3 taskset -c 64-79"
L=/disk3/trbench/logs; OUT=results/cpu_speed.jsonl
if pgrep -f "llama-server|eval.run" >/dev/null; then
  echo "[speed] baska bir olcum calisiyor; ayni cekirdekleri paylasirsa iki olcum de bozulur" >&2; exit 1
fi
if [ ! -x "$LCPP/build/bin/llama-bench" ]; then
  $LOW cmake --build "$LCPP/build" -j 16 --target llama-bench > "$L/llama_bench_build.log" 2>&1
fi
: > "$OUT"
for F in /disk3/trbench/gguf/*/*.gguf; do
  case "$F" in *-0000[2-9]-of-*) continue ;; esac   # bolunmus dosyada sadece ilk parca
  echo "[speed] $(basename "$F") $(date)"
  $LOW "$LCPP/build/bin/llama-bench" -m "$F" -t 16 -p 512 -n 128 -r 3 -o jsonl >> "$OUT" 2>> "$L/llama_bench.log"
done
echo "[speed] bitti $(date): $(wc -l < "$OUT") satir"
