#!/usr/bin/env bash
# GPU olcumleri icin kurulum. GPU KULLANMAZ (derleme + paket kurulumu), servis calisirken yapilabilir.
#   bash scripts/gpu_setup.sh
#  - llama.cpp b12470, CUDA, V100 (SM70) -> /disk3/trbench/llama.cpp/build-cuda
#  - vLLM: sunucudaki V100 derlemesi (wheel yolu env.local'da: VLLM_WHEEL) -> /disk3/trbench/venv-vllm
set -euo pipefail
trap 'echo "[setup] HATA: satir $LINENO, komut: $BASH_COMMAND" >&2' ERR
[ -f /disk3/trbench/env.local ] && source /disk3/trbench/env.local
R=/disk3/trbench; L=$R/logs; mkdir -p "$L" "$R/tmp"
export TMPDIR=$R/tmp CUDA_VISIBLE_DEVICES=""
LOW="nice -n 19 ionice -c 3 taskset -c 64-79"
LLAMA_TAG=b12470; LCPP=$R/llama.cpp

[ -d "$LCPP" ] || git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$LCPP"
command -v nvcc >/dev/null || export PATH=/usr/local/cuda/bin:$PATH
nvcc --version | tail -1
if [ ! -x "$LCPP/build-cuda/bin/llama-server" ]; then
  echo "[setup] llama.cpp CUDA (SM70) derleniyor $(date)  (20-40 dk)"
  $LOW cmake -S "$LCPP" -B "$LCPP/build-cuda" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
      > "$L/llama_cuda_build.log" 2>&1
  $LOW cmake --build "$LCPP/build-cuda" -j 16 --target llama-server llama-bench >> "$L/llama_cuda_build.log" 2>&1
fi
ls -l "$LCPP/build-cuda/bin/llama-server" "$LCPP/build-cuda/bin/llama-bench"

WHL=${VLLM_WHEEL:?env.local icinde VLLM_WHEEL tanimli olmali (sunucuya ozel yol, repoda tutulmaz)}
if [ ! -x "$R/venv-vllm/bin/vllm" ]; then
  echo "[setup] vLLM kuruluyor: $WHL $(date)"
  python3 -m venv "$R/venv-vllm"
  $LOW "$R/venv-vllm/bin/pip" install -q "$WHL" > "$L/vllm_install.log" 2>&1
fi
"$R/venv-vllm/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
echo "[setup] bitti $(date)"
