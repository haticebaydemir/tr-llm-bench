#!/usr/bin/env bash
# Model kuyrugu (CPU): her model icin GGUF uret -> sablon kontrolu -> kalite olcumu.
# Bir model basarisiz olursa digerleri devam eder; sablonu olmayan model OLCULMEZ
# (18 Kasim Trendyol dersi: llama.cpp sessizce ChatML'e duser, sonuclar anlamsiz olur).
#   bash scripts/queue_models.sh "HF_REPO kisa-ad" "HF_REPO kisa-ad" ...
set -uo pipefail
L=/disk3/trbench/logs
FAIL=0   # Faz 7r: atlanan ya da hata veren model varsa son satir HATA, cikis kodu 1
for pair in "$@"; do
  set -- $pair; HF=$1; NAME=$2
  echo "[queue] $NAME basliyor $(date)"
  if ! bash scripts/make_gguf.sh "$HF" "$NAME"; then
    echo "[queue] $NAME: GGUF uretilemedi, atlaniyor"; FAIL=1; continue
  fi
  if ! grep -q "chat_template" "$L/convert_$NAME.log"; then
    echo "[queue] $NAME: sohbet sablonu YOK, olculmedi (elle sablon verilmeli)"; FAIL=1; continue
  fi
  bash scripts/phase4_cpu.sh "/disk3/trbench/gguf/$NAME" "$NAME" || { echo "[queue] $NAME: olcum hatasi"; FAIL=1; }
done
if [ "$FAIL" -ne 0 ]; then
  echo "[queue] HATA: en az bir model olculemedi $(date)"
  exit 1
fi
echo "[queue] bitti $(date)"
