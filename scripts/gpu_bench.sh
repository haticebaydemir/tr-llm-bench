#!/usr/bin/env bash
# GPU 2 olcumu (llama.cpp ya da vLLM), guard ile.
#   nohup bash scripts/gpu_bench.sh configs/gpu_probe.yaml > /disk3/trbench/logs/gpu_probe.out 2>&1 &
# Yarida kalirsa ayni komut: bitmis kosular atlanir.
set -euo pipefail
trap 'echo "[gpu] HATA: satir $LINENO, komut: $BASH_COMMAND" >&2' ERR
CFG=${1:?config}
cd /disk3/trbench/tr-llm-bench
[ -f /disk3/trbench/env.local ] && source /disk3/trbench/env.local
export HF_HOME=/disk3/trbench/hf TMPDIR=/disk3/trbench/tmp
PY=/disk3/trbench/venv/bin/python
L=/disk3/trbench/logs; mkdir -p "$L"

$PY -m trbench.config "$CFG"                          # butceyi asan config burada durur
[ -f data/eval/eval_set.jsonl ] || $PY -m trbench.eval.build | tail -1
$PY -m trbench.guard --preflight                      # sunucu uygun degilse burada durur

KILL_BENCH="pkill -f trbench.runner; pkill -f 'alias trbench-'; pkill -f 'served-model-name trbench-'"
# pkill -f YOK: komut satirinda bu yaziyi gecen her sureci (bu scripti cagiran zinciri de) oldurur.
GUARD=""
cleanup() {
  [ -n "$GUARD" ] && kill "$GUARD" 2>/dev/null || true
  pkill -f 'alias trbench-' || true
  pkill -f 'served-model-name trbench-' || true
  sleep 8; echo "[gpu] TEMIZLENDI $(date)"; nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader
}
trap cleanup EXIT
trap 'exit 143' INT TERM

nohup $PY -m trbench.guard --stop-cmd "$KILL_BENCH" > "$L/gpu_guard.log" 2>&1 &
GUARD=$!
sleep 3
kill -0 "$GUARD" 2>/dev/null || { echo "[gpu] guard baslamadi, olcum yapilmiyor" >&2; exit 1; }
echo "[gpu] basladi $CFG $(date)"
$PY -m trbench.runner "$CFG" &
RUNNER=$!
# Bekci: guard herhangi bir nedenle olurse olcum korumasiz kalmasin -> durdur.
while kill -0 "$RUNNER" 2>/dev/null; do
  if ! kill -0 "$GUARD" 2>/dev/null; then
    echo "[gpu] GUARD DURDU -> olcum de durduruluyor $(date)" >&2
    eval "$KILL_BENCH" || true
    exit 1
  fi
  sleep 5
done
wait "$RUNNER"
echo "[gpu] bitti $(date)"
