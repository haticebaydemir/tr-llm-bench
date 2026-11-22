#!/usr/bin/env bash
# Olcum ortaminin kilidi -> env/ (repoya girer). GPU kullanmaz.
#   cd /disk3/trbench/yeni && bash scripts/env_snapshot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
R=/disk3/trbench; mkdir -p env
"$R/venv/bin/pip" freeze --exclude-editable | sort > env/requirements-server.lock
[ -x "$R/venv-vllm-up/bin/pip" ] && "$R/venv-vllm-up/bin/pip" freeze | sort > env/requirements-vllm-upstream.lock
{
  echo "# Olcum ortami ($(date -u +%F))"
  echo
  echo "- OS: $(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r)"
  echo "- CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs), $(nproc) cekirdek"
  echo "- RAM: $(awk '/MemTotal/ {printf "%.0f GB", $2/1024/1024}' /proc/meminfo)"
  echo "- GPU: $(nvidia-smi -i 2 --query-gpu=name,driver_version --format=csv,noheader)"
  echo "- CUDA (surucu): $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | cut -d' ' -f3)"
  NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
  echo "- CUDA toolkit (llama.cpp derlemesi): $("$NVCC" --version 2>/dev/null | tail -1 || echo bulunamadi)"
  echo "- llama.cpp: tag b12470, commit $(git -C "$R/llama.cpp" rev-parse --short HEAD), CUDA build SM70"
  echo "- Python (olcum venv): $("$R/venv/bin/python" -V 2>&1)"
  [ -x "$R/venv-vllm-up/bin/python" ] && echo "- vLLM upstream: $("$R/venv-vllm-up/bin/python" -c 'import vllm,torch;print(vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)')"
  echo "- vLLM AWQ sutunu: V100 icin yamanmis bir derleme (public degil; README 2b)"
} > env/SYSTEM.md
cat env/SYSTEM.md
wc -l env/*.lock
