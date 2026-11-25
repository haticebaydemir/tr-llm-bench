"""Trendyol tokenizer vergisi neden 1.10 / 1.12 / 1.22? (CLAIMS.md T8)

  python -m trbench.tokenizer_tax.trendyol_versions

Ayni tokenizer dosyasi (sabit revision), ayni FLORES+ cumleleri; degisen yalnizca kutuphane
surumu ve fast/slow tokenizer. Her calistirma results/tokenizer_trendyol.csv'ye bir satir
ekler (surum, sinif, vocab, TR/EN orani); farkli transformers surumlerinde (ayri venv)
tekrar calistirilarak tablo dolar. Metin yazilmaz, yalnizca sayilar.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from trbench.tokenizer_tax.corpus import DATA_DIR

REPO = "Trendyol/Trendyol-LLM-7b-chat-v1.8"
OUT = Path("results/tokenizer_trendyol.csv")
FIELDS = ["transformers", "tokenizers", "revision", "use_fast", "class", "vocab", "tr_tokens", "en_tokens",
          "ratio", "error"]


def measure(texts: pd.DataFrame, use_fast: bool) -> dict:
    import tokenizers
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    rev = HfApi().model_info(REPO).sha
    row = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
           "revision": rev, "use_fast": use_fast}
    try:
        tok = AutoTokenizer.from_pretrained(REPO, revision=rev, use_fast=use_fast)
    except Exception as e:  # noqa: BLE001 - yuklenemeyen sinif da bir sonuc
        return {**row, "error": f"{type(e).__name__}: {str(e)[:80]}"}
    n = {lang: sum(len(x) for x in tok(texts[lang].tolist(), add_special_tokens=False)["input_ids"])
         for lang in ("eng_Latn", "tur_Latn")}
    return {**row, "class": type(tok).__name__, "vocab": len(tok), "tr_tokens": n["tur_Latn"],
            "en_tokens": n["eng_Latn"], "ratio": round(n["tur_Latn"] / n["eng_Latn"], 4)}


def main() -> None:
    texts = pd.read_parquet(DATA_DIR / "flores_parallel.parquet")
    new = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for fast in (True, False):
            row = measure(texts, fast)
            w.writerow(row)
            print(row)


if __name__ == "__main__":
    main()
