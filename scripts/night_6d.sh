#!/usr/bin/env bash
# Faz 6d gecesi (GPU 2, ~8 GB pay, guard her adimda acik). Siralama: genis butceden dara.
#   cd /disk3/trbench/yeni && nohup bash scripts/night_6d.sh > /disk3/trbench/logs/night_6d.out 2>&1 &
# Her adim bagimsiz: biri basarisiz olursa digeri calisir. Servis etkilenirse guard o adimi durdurur.
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0   # Faz 7r: bir adim hata verirse rapor uretilmez, son satir HATA olur, cikis kodu 1
export TRBENCH_VLLM=/disk3/trbench/venv-vllm-up/bin/vllm
echo "[6d] basladi $(date)"
for c in gpu_6d_util20 gpu_6d_matched gpu_6d_util19 gpu_6d_util18; do
  bash scripts/gpu_bench.sh "configs/$c.yaml" || { echo "[6d] $c hata"; FAIL=1; }
done
unset TRBENCH_VLLM
bash scripts/gpu_bench.sh configs/gpu_6d_samples.yaml || { echo "[6d] samples hata"; FAIL=1; }
if [ "$FAIL" -ne 0 ]; then
  echo "[6d] HATA: en az bir adim basarisiz, rapor uretilmedi $(date)"
  exit 1
fi
echo "[6d] HEPSI BITTI $(date)"
