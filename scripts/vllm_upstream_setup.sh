#!/usr/bin/env bash
# Upstream vLLM (PyPI) ayri venv'e. GPU KULLANMAZ. 0.20+ V100'u (SM70) desteklemiyor -> 0.18.x.
set -euo pipefail
R=/disk3/trbench; V=$R/venv-vllm-up; L=$R/logs; mkdir -p "$L" "$R/tmp"
export TMPDIR=$R/tmp PIP_CACHE_DIR=$R/tmp/pip CUDA_VISIBLE_DEVICES=""
LOW="nice -n 19 ionice -c 3 taskset -c 64-79"
if [ ! -x "$V/bin/vllm" ]; then
  echo "[up] kuruluyor $(date)"
  python3 -m venv "$V"
  $LOW "$V/bin/pip" install -q "vllm>=0.18,<0.19" > "$L/vllm_up_install.log" 2>&1
fi
"$V/bin/python" -c "import vllm, torch; print('[up] vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
