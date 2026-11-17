#!/usr/bin/env bash
# Orijinal HF agirliklarindan GGUF uretir: fp16 + q8_0 + q4_k_m.
# Ucuncu kisilerin GGUF'lari yerine: her model ayni llama.cpp surumu ve ayni ayarla.
# SADECE CPU, 16 cekirdek, en dusuk oncelik.
#   bash scripts/make_gguf.sh Trendyol/Trendyol-LLM-7b-chat-v1.8 trendyol-7b
# Cikti: /disk3/trbench/gguf/<ad>/<ad>-{fp16,q8_0,q4_k_m}.gguf
set -euo pipefail
trap 'echo "[gguf] HATA: satir $LINENO, komut: $BASH_COMMAND" >&2' ERR
HF=${1:?hf repo}; NAME=${2:?kisa ad}
[ -f /disk3/trbench/env.local ] && source /disk3/trbench/env.local
export HF_HOME=/disk3/trbench/hf TMPDIR=/disk3/trbench/tmp CUDA_VISIBLE_DEVICES=""
mkdir -p "$TMPDIR"
LLAMA_TAG=b12470; LCPP=/disk3/trbench/llama.cpp
LOW="nice -n 19 ionice -c 3 taskset -c 64-79"
L=/disk3/trbench/logs; mkdir -p "$L"
OUT=/disk3/trbench/gguf/$NAME; mkdir -p "$OUT"
SRC=/disk3/trbench/hf_src/$NAME

[ -d "$LCPP" ] || git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$LCPP"
if [ ! -x "$LCPP/build/bin/llama-quantize" ] || [ ! -x "$LCPP/build/bin/llama-server" ]; then
  echo "[gguf] llama.cpp $LLAMA_TAG derleniyor $(date)"
  $LOW cmake -S "$LCPP" -B "$LCPP/build" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=ON > "$L/llama_build.log" 2>&1
  $LOW cmake --build "$LCPP/build" -j 16 --target llama-server llama-quantize >> "$L/llama_build.log" 2>&1
fi

# Donusturucu kendi bagimliliklarini (torch CPU) ister: ana venv'e karismasin.
CV=/disk3/trbench/venv-convert
if [ ! -x "$CV/bin/python" ]; then
  echo "[gguf] donusturucu ortami kuruluyor $(date)"
  python3 -m venv "$CV"
  $LOW "$CV/bin/pip" install -q -r "$LCPP/requirements/requirements-convert_hf_to_gguf.txt"
fi

F16=$OUT/$NAME-fp16.gguf
if [ ! -f "$F16" ]; then
  echo "[gguf] $HF indiriliyor $(date)"
  $LOW /disk3/trbench/venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF', local_dir='$SRC',
    allow_patterns=['*.json','*.safetensors','*.model','*.tiktoken','*.txt'])"
  echo "[gguf] fp16'ya donusturuluyor $(date)"
  $LOW "$CV/bin/python" "$LCPP/convert_hf_to_gguf.py" "$SRC" --outtype f16 --outfile "$F16.tmp" \
      > "$L/convert_$NAME.log" 2>&1
  mv "$F16.tmp" "$F16"
  rm -rf "$SRC"   # 15 GB yer acar; gerekirse yeniden indirilir
fi
for Q in q8_0 q4_k_m; do
  F=$OUT/$NAME-$Q.gguf
  [ -f "$F" ] && continue
  echo "[gguf] $Q $(date)"
  $LOW "$LCPP/build/bin/llama-quantize" "$F16" "$F.tmp" "${Q^^}" 16 > "$L/quantize_${NAME}_$Q.log" 2>&1
  mv "$F.tmp" "$F"
done
ls -lh "$OUT"
echo "[gguf] bitti $(date)"
