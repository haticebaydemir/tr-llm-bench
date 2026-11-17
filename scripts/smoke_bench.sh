#!/usr/bin/env bash
# Duman testi: 0.5B model, GPU 2, TR/EN x esanlilik 1/4. Guard acik, cikista temizlik.
# Kullanim (sunucuda): bash scripts/smoke_bench.sh
set -euo pipefail
cd /disk3/trbench/tr-llm-bench
[ -f /disk3/trbench/env.local ] && source /disk3/trbench/env.local   # sunucuya ozel ayarlar (repo disi)
export HF_HOME=/disk3/trbench/hf TMPDIR=/disk3/trbench/tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/disk3/trbench/venv/bin/python
L=/disk3/trbench/logs; mkdir -p "$L"

$PY -m trbench.eval.build | tail -3
$PY -m trbench.guard --preflight

cleanup() {
  pkill -f trbench.guard || true
  [ -f "$L/smoke_vllm.pid" ] && kill "$(cat "$L/smoke_vllm.pid")" 2>/dev/null || true
  sleep 8; echo "TEMIZLENDI"
}
trap cleanup EXIT

nohup $PY -m trbench.guard --stop-cmd "kill \$(cat $L/smoke_vllm.pid)" > "$L/bench_guard.log" 2>&1 &
sleep 3
nohup /disk3/trbench/venv-vllm/bin/vllm serve /disk3/trbench/models/qwen2.5-0.5b-instruct \
  --served-model-name smoke --host 127.0.0.1 --port 18000 --dtype half \
  --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.30 > "$L/smoke_vllm.log" 2>&1 &
echo $! > "$L/smoke_vllm.pid"
timeout 300 bash -c 'until curl -sf 127.0.0.1:18000/health >/dev/null; do sleep 5; done'

for c in 1 4; do for lang in tr en; do
  $PY -m trbench.load.generator --model smoke --concurrency $c --n 16 \
      --lang $lang --max-tokens 256 > "$L/smoke_${lang}_c${c}.json"
  $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"$lang c=$c | hata {d['errors']} | TTFT p50 {d['ttft_p50_ms']:.0f} ms | {d['tokens_per_s']:.0f} tok/s | {d['words_per_s']:.0f} kelime/s\")" "$L/smoke_${lang}_c${c}.json"
done; done
cat "$L/bench_guard.log"
