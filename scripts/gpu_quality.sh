#!/usr/bin/env bash
# Dogruluk olcumu, GPU 2, llama.cpp CUDA, Q4_K_M. Guard acik; diger GPU'lara ve servise dokunmaz.
#   nohup bash scripts/gpu_quality.sh configs/gpu_quality.txt > /disk3/trbench/logs/gpu_quality.out 2>&1 &
# Yarida kalirsa ayni komut: bitmis sorular atlanir. Bir model basarisiz olursa digerleri devam eder.
set -uo pipefail
LIST=${1:?model listesi}
[ -f /disk3/trbench/env.local ] && source /disk3/trbench/env.local
export HF_HOME=/disk3/trbench/hf TMPDIR=/disk3/trbench/tmp CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/disk3/trbench/venv/bin/python
L=/disk3/trbench/logs; mkdir -p "$L" results
SERVER=/disk3/trbench/llama.cpp/build-cuda/bin/llama-server
PORT=18010; OUT=${OUT:-results/gpu_quality.jsonl}; MAX_GB=6.2

$PY -m trbench.guard --preflight || { echo "[gq] on kontrol gecmedi, baslamiyor"; exit 1; }
GUARD=""
cleanup() {
  pkill -f "llama-server.*--port $PORT" 2>/dev/null
  pkill -f "trbench.eval.run --url http://127.0.0.1:$PORT" 2>/dev/null
  [ -n "$GUARD" ] && kill "$GUARD" 2>/dev/null
  sleep 3; echo "[gq] TEMIZLENDI $(date)"
}
trap cleanup EXIT
trap 'exit 143' INT TERM
nohup $PY -m trbench.guard --stop-cmd "pkill -f 'trbench.eval.run --url http://127.0.0.1:$PORT'; pkill -f 'llama-server.*--port $PORT'" \
  > "$L/gq_guard.log" 2>&1 &
GUARD=$!
sleep 3
kill -0 "$GUARD" 2>/dev/null || { echo "[gq] guard baslamadi"; exit 1; }

while read -r REPO NAME TPL MORE <&3; do
  kill -0 "$GUARD" 2>/dev/null || { echo "[gq] GUARD DURDU, olcum bitiriliyor"; break; }
  echo "[gq] $NAME basliyor $(date)"
  GGUF=$($PY -m trbench.eval.gguf "$REPO" q4_k_m 2>>"$L/gq_download.log" | tail -1)
  if [ ! -f "$GGUF" ]; then echo "[gq] $NAME: GGUF bulunamadi/indirilemedi, atlaniyor"; continue; fi
  SIZE=$(du -bc "$(dirname "$GGUF")"/"$(basename "$GGUF" | sed 's/-00001-of-.*//')"* | tail -1 | cut -f1)
  if awk -v s="$SIZE" -v m="$MAX_GB" 'BEGIN{exit !(s > m*1024^3)}'; then echo "[gq] $NAME: $SIZE bayt, butceye sigmaz, atlaniyor"; continue; fi
  if ss -ltn | grep -q ":$PORT "; then echo "[gq] HATA: port $PORT dolu"; break; fi
  # 3. sutun: sablon dosyasi ya da '-'; sonrasi llama-server'a aynen gecer
  EXTRA="${MORE:-}"; [ -n "${TPL:-}" ] && [ "$TPL" != "-" ] && EXTRA="--chat-template-file $TPL $EXTRA"
  # --reasoning-budget 0: Qwen3 gibi modellerde <think> kapali (diger modellerde etkisiz)
  "$SERVER" -m "$GGUF" --host 127.0.0.1 --port $PORT -ngl 99 -c 8192 -np 4 -t 8 \
      --reasoning-budget 0 $EXTRA > "$L/gq_${NAME}.log" 2>&1 &
  SPID=$!
  if ! timeout 600 bash -c "until curl -sf 127.0.0.1:$PORT/health >/dev/null; do kill -0 $SPID || exit 1; sleep 3; done"; then
    echo "[gq] $NAME: sunucu acilmadi (log: $L/gq_${NAME}.log)"; kill $SPID 2>/dev/null; SPID=""; continue
  fi
  echo "[gq] $NAME GPU2: $(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader) | $(grep -m1 -o 'KV.*size *= *[0-9.]* MiB.*' "$L/gq_${NAME}.log")"
  $PY -m trbench.eval.run --url http://127.0.0.1:$PORT --label "$NAME/q4_k_m" --parallel 4 --out "$OUT" \
      | tail -12
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; SPID=""
  sleep 5
done 3< <(grep -v '^\s*#' "$LIST" | grep -v '^\s*$')
echo "[gq] HEPSI BITTI $(date)"
