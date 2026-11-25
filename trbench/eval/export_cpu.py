"""CPU sonuclarini repoya konacak hale getirir (CLAIMS.md acik madde 1).

  python -m trbench.eval.export_cpu            # kontrol + results/phase4_scores.jsonl
  python -m trbench.eval.export_cpu --check    # yalnizca kontrol, dosya yazmaz

results/phase4.jsonl modelin cevap metnini (response) tasiyor. Cevaplar ucuncu taraf soru
metnini (Belebele pasaji, GSM8K sorusu) tekrar edebilir; bu yuzden yayinlanan dosyada
response alani YOK. Analizler yalnizca correct/error/label/id/lang/category kullaniyor,
dolayisiyla tablolar ayni cikar (asagida dogrulanir).

results/cpu_speed.jsonl (llama-bench hizlari) yalnizca sayi icermeli; metin alani varsa durur.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FULL = Path("results/phase4.jsonl")
OUT = Path("results/phase4_scores.jsonl")
SPEED = Path("results/cpu_speed.jsonl")
DROP = ("response", "raw_response")  # model cevaplari (ham ve temizlenmis)
TEXT_LIMIT = 80  # bundan uzun metin alani = muhtemel soru/cevap metni
SAFE = ("id", "label", "model_filename")  # id: kaynak URL'si (Belebele); model_filename: GGUF yolu


def strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in DROP}


def long_text_fields(row: dict) -> list[str]:
    return [k for k, v in row.items() if k not in SAFE and isinstance(v, str) and len(v) > TEXT_LIMIT]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(x) for x in FULL.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = [strip(r) for r in rows]
    bad = sorted({k for r in out for k in long_text_fields(r)})
    print(f"{FULL}: {len(rows)} satir, alanlar: {sorted({k for r in rows for k in r})}")
    print(f"cikarilan alanlar: {list(DROP)}; kalan uzun metin alanlari: {bad or 'yok'}")

    if SPEED.exists():
        sp = [json.loads(x) for x in SPEED.read_text(encoding="utf-8").splitlines() if x.strip()]
        sbad = sorted({k for r in sp for k in long_text_fields(r)})
        print(f"{SPEED}: {len(sp)} satir, alanlar: {sorted({k for r in sp for k in r})}; "
              f"uzun metin: {sbad or 'yok'}")
    else:
        sbad = []
        print(f"{SPEED}: yok")

    if bad or sbad:
        raise SystemExit("DUR: uzun metin alani var, yayinlamadan once bak.")
    if a.check:
        return
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"yazildi: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
