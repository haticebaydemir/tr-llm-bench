#!/usr/bin/env bash
# Faz 7t: baglam penceresi vergisi (GPU 2, ~7,5 GB pay, guard acik), ~1 saat.
# Her model icin llama-bench: ayni icerik Ingilizce (E token) ve Turkce (E x vergi token) uzunlukta
#   - uretim hizi o derinlikte (-p 0 -n 64 -d ...)  - prefill hizi o uzunlukta (-p ... -n 0)
#   cd /disk3/trbench/yeni && nohup bash scripts/gpu_context.sh > /disk3/trbench/logs/gpu_context.out 2>&1 &
# Hata olursa rapor uretilmez, son satir HATA olur, cikis kodu 1 (Faz 7r).
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0
[ -f /disk3/trbench/env.local ] && source /disk3/trbench/env.local
export HF_HOME=/disk3/trbench/hf TMPDIR=/disk3/trbench/tmp CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/disk3/trbench/venv/bin/python
B=/disk3/trbench/llama.cpp/build-cuda/bin/llama-bench
L=/disk3/trbench/logs; mkdir -p "$L" results
OUT=results/gpu_context.jsonl

$PY -m trbench.guard --preflight || { echo "[ctx] on kontrol gecmedi, baslamiyor"; exit 1; }
if pgrep -f "llama-server|llama-bench|trbench.runner" >/dev/null; then echo "[ctx] baska olcum calisiyor"; exit 1; fi
GUARD=""
cleanup() { pkill -f "llama-bench" 2>/dev/null; [ -n "$GUARD" ] && kill "$GUARD" 2>/dev/null; sleep 2; echo "[ctx] TEMIZLENDI $(date)"; }
trap cleanup EXIT
trap 'exit 143' INT TERM
nohup $PY -m trbench.guard --stop-cmd "pkill -f llama-bench" > "$L/ctx_guard.log" 2>&1 &
GUARD=$!
sleep 3
kill -0 "$GUARD" 2>/dev/null || { echo "[ctx] guard baslamadi"; exit 1; }

echo "[ctx] basladi $(date)"
: > "$OUT"
while read -r REPO NAME DEPTHS <&3; do
  kill -0 "$GUARD" 2>/dev/null || { echo "[ctx] GUARD DURDU"; FAIL=1; break; }
  GGUF=$($PY -m trbench.eval.gguf "$REPO" q4_k_m 2>>"$L/ctx_download.log" | tail -1)
  if [ ! -f "$GGUF" ]; then echo "[ctx] $NAME: GGUF yok"; FAIL=1; continue; fi
  echo "[ctx] $NAME derinlikler $DEPTHS $(date)"
  "$B" -m "$GGUF" -ngl 99 -p 0 -n 64 -d "$DEPTHS" -r 3 -o jsonl >> "$OUT" 2>> "$L/ctx_bench.log" \
    || { echo "[ctx] $NAME: uretim olcumu hata"; FAIL=1; continue; }
  "$B" -m "$GGUF" -ngl 99 -p "$DEPTHS" -n 0 -r 3 -o jsonl >> "$OUT" 2>> "$L/ctx_bench.log" \
    || { echo "[ctx] $NAME: prefill olcumu hata"; FAIL=1; continue; }
  echo "[ctx] $NAME GPU2 $(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader)"
done 3< <($PY -m trbench.eval.context_tax --plan)
if [ "$FAIL" -ne 0 ]; then
  echo "[ctx] HATA: en az bir adim basarisiz, rapor uretilmedi $(date)"
  exit 1
fi
$PY -m trbench.eval.context_tax
$PY -m trbench.eval.context_tax --content 1024
echo "[ctx] HEPSI BITTI $(date)"
