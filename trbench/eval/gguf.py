"""GGUF dosyalarini bulur ve indirir.

Yollari tahmin etmiyoruz (FLORES dersi): repo dosya listesine bakip secer.
llama.cpp ise scripts/phase4_cpu.sh icinde SABIT bir surumden derlenir
(hazir ikili arama 17 Kasim'da kirildi: 'latest' surumu artik ikili icermiyor).

Kullanim (sunucuda):
  python -m trbench.eval.gguf Qwen/Qwen2.5-7B-Instruct-GGUF q4_k_m
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path("/disk3/trbench")


def pick_files(files: list[str], quant: str) -> list[str]:
    """'q4_k_m' icin tek dosya ya da bolunmus parcalarin hepsi (sirali).
    Buyuk/kucuk harf ve '-'/'_' farki onemsiz: Q4_K_M, q4-k-m ayni."""
    norm = quant.lower().replace("-", "_")
    hits = [f for f in files if f.endswith(".gguf")
            and re.search(rf"[-._]{re.escape(norm)}([-._]|$)", f.lower().replace("-", "_"))]
    return sorted(hits)


def first_part(paths: list[str]) -> str:
    """Bolunmus GGUF'ta llama.cpp'ye ilk parca verilir (-00001-of-0000N)."""
    return next((p for p in paths if "00001-of" in p), paths[0])


def download_model(repo: str, quant: str) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    parts = pick_files(HfApi().list_repo_files(repo), quant)
    if not parts:
        raise SystemExit(f"{repo}: '{quant}' bulunamadi")
    local = ROOT / "gguf" / repo.replace("/", "__")
    paths = [hf_hub_download(repo, p, local_dir=local) for p in parts]
    return Path(first_part(paths))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("quant")
    a = ap.parse_args()
    print(download_model(a.repo, a.quant))


if __name__ == "__main__":
    main()
