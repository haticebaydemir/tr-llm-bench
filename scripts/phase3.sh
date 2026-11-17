#!/usr/bin/env bash
# Faz 3 performans matrisi. Gece calistir:
#   nohup bash scripts/phase3.sh > /disk3/trbench/logs/phase3.out 2>&1 &
# Yarida kalirsa ayni komutla yeniden baslat: bitmis kosular atlanir.
set -euo pipefail
cd /disk3/trbench/tr-llm-bench
[ -f /disk3/trbench/env.local ] && source /disk3/trbench/env.local   # sunucuya ozel ayarlar (repo disi)
export HF_HOME=/disk3/trbench/hf TMPDIR=/disk3/trbench/tmp
PY=/disk3/trbench/venv/bin/python
L=/disk3/trbench/logs; mkdir -p "$L"

$PY -m trbench.config configs/phase3.yaml           # config gecersizse burada durur
[ -f data/eval/eval_set.jsonl ] || $PY -m trbench.eval.build | tail -1
$PY -m trbench.guard --preflight                     # sunucu uygun degilse burada durur

KILL_BENCH="pkill -f trbench.runner; pkill -f 'served-model-name trbench-'"
cleanup() {
  pkill -f trbench.guard || true
  pkill -f 'served-model-name trbench-' || true
  sleep 8; echo "[phase3] TEMIZLENDI $(date)"
}
trap cleanup EXIT

nohup $PY -m trbench.guard --stop-cmd "$KILL_BENCH" > "$L/phase3_guard.log" 2>&1 &
sleep 3
echo "[phase3] basladi $(date)"
$PY -m trbench.runner configs/phase3.yaml
echo "[phase3] bitti $(date)"
