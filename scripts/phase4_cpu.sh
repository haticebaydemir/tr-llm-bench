#!/usr/bin/env bash
# Faz 4 kalite olcumu: llama.cpp, SADECE CPU (GPU'ya dokunmaz).
# 16 cekirdek (64-79), en dusuk oncelik (nice 19, ionice idle). Guard paylasilan servisi izler.
#   nohup bash scripts/phase4_cpu.sh Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b \
#       > /disk3/trbench/logs/phase4.out 2>&1 &
# Yarida kalirsa ayni komut: bitmis sorular atlanir.
# GGUF'ta sohbet sablonu yoksa (18 Kasim, Trendyol) llama.cpp ChatML'e duser; dogru sablonu ver:
#   LLAMA_EXTRA="--chat-template-file configs/chat_templates/mistral_inst.jinja" bash scripts/phase4_cpu.sh ...
# Dikkat: "--chat-template llama2" gibi bir AD vermek bu llama.cpp surumunde sablonun
# KENDISI sayilir (Jinja) ve modele sadece "llama2" kelimesi gider. Her zaman dosya ver.
set -euo pipefail
# set -e sessiz cikar; nerede durdugunu her zaman yazdir.
trap 'echo "[phase4] HATA: satir $LINENO, komut: $BASH_COMMAND" >&2' ERR
REPO=${1:?gguf repo}; NAME=${2:?kisa ad}; QUANTS=${3:-"q4_k_m q8_0 fp16"}
cd /disk3/trbench/tr-llm-bench
[ -f /disk3/trbench/env.local ] && source /disk3/trbench/env.local   # sunucuya ozel ayarlar (repo disi)
export HF_HOME=/disk3/trbench/hf TMPDIR=/disk3/trbench/tmp CUDA_VISIBLE_DEVICES=""
PY=/disk3/trbench/venv/bin/python
L=/disk3/trbench/logs; mkdir -p "$L"
CORES=64-79; PORT=18010
# llama.cpp SABIT surum: bir yil sonra calistiran da ayni kodu derler.
LLAMA_TAG=b12470; LCPP=/disk3/trbench/llama.cpp
LOW="nice -n 19 ionice -c 3 taskset -c $CORES"

[ -f data/eval/eval_set.jsonl ] || $PY -m trbench.eval.build | tail -1
$PY -m trbench.guard --preflight
SERVER=$LCPP/build/bin/llama-server
if [ ! -x "$SERVER" ]; then
  echo "[phase4] llama.cpp $LLAMA_TAG derleniyor (16 cekirdek, dusuk oncelik) $(date)"
  rm -rf "$LCPP"
  git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$LCPP"
  $LOW cmake -S "$LCPP" -B "$LCPP/build" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DLLAMA_USE_PREBUILT_UI=OFF > "$L/llama_build.log" 2>&1
  $LOW cmake --build "$LCPP/build" -j 16 --target llama-server llama-quantize >> "$L/llama_build.log" 2>&1
fi
echo "[phase4] llama.cpp: $LLAMA_TAG $SERVER"

cleanup() {
  pkill -f trbench.eval.run || true
  pkill -f "llama-server.*--port $PORT" || true
  pkill -f trbench.guard || true
  sleep 3; echo "[phase4] TEMIZLENDI $(date)"
}
# 17 Kasim: sadece EXIT yetmedi. Bash on plandaki komut bitene kadar sinyali islemez;
# script durduruldu ama llama-server ve eval.run calismaya devam etti.
# Cozum: sinyalleri yakala + uzun isleri arka planda baslatip 'wait' ile bekle.
trap cleanup EXIT
trap 'exit 143' INT TERM
nohup $PY -m trbench.guard --stop-cmd "pkill -f trbench.eval.run; pkill -f 'llama-server.*--port $PORT'" \
  > "$L/phase4_guard.log" 2>&1 &
sleep 3

for Q in $QUANTS; do
  GGUF=$($LOW $PY -m trbench.eval.gguf "$REPO" "$Q" | tail -1)
  echo "[phase4] $NAME/$Q -> $GGUF  $(date)"
  # Port bos olmali: 17 Kasim'da eski Q8 sunucusu portu tutuyordu ve yeni olcum
  # "q4" etiketiyle ona baglandi (yanlis etiketli veri).
  if ss -ltn | grep -q ":$PORT "; then
    echo "[phase4] HATA: port $PORT dolu, baska bir sunucu calisiyor" >&2; exit 1
  fi
  $LOW "$SERVER" -m "$GGUF" --host 127.0.0.1 --port $PORT -t 16 -c 8192 -np 4 ${LLAMA_EXTRA:-} \
       > "$L/llama_${NAME}_${Q}.log" 2>&1 &
  SPID=$!
  timeout 900 bash -c "until curl -sf 127.0.0.1:$PORT/health >/dev/null; do sleep 5; done"
  kill -0 $SPID   # cevap veren sunucu BIZIM baslattigimiz mi?
  curl -s 127.0.0.1:$PORT/props | grep -qi "$Q" \
    || { echo "[phase4] HATA: sunucudaki model $Q degil" >&2; exit 1; }
  $LOW $PY -m trbench.eval.run --url http://127.0.0.1:$PORT --label "$NAME/$Q" --parallel 4 &
  wait $!
  kill $SPID; wait $SPID 2>/dev/null || true
done
echo "[phase4] bitti $(date)"
