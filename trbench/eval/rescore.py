"""Trendyol duzeltmesi (18 Kasim): model '<|im_end|>' turn sonunu DURMA isareti olarak
degil, METIN olarak uretti; llama.cpp durmadi ve cevap tekrar eden sahte turlarla doldu
('<|im_end|>user ...'). Bizim GGUF'umuzda modelin bitis token'i sohbet sonu olarak
isaretlenmemis.

Greedy (temperature 0) uretim belirleyici oldugu icin, cevabi ilk '<|im_end|>'da kesmek
dogru durma ayariyla alinacak cevabin AYNISIDIR. Bu yuzden olcumu tekrarlamadan,
cevaplari keserek yeniden puanliyoruz. Orijinal cevap 'raw_response' alaninda kalir.

Kullanim:  python -m trbench.eval.rescore trendyol-7b
           python -m trbench.eval.rescore aya-expanse-8b --file results/gpu_quality.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from trbench.eval.run import OUT, load_items
from trbench.eval.scorers import score

STOPS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|END_OF_TURN_TOKEN|>")  # son: Aya (Cohere)


def cut(text: str) -> str:
    """Ilk turn-sonu isaretinde kes (dogru durma ayarinin verecegi cevap)."""
    for s in STOPS:
        text = text.split(s, 1)[0]
    return text


def rescore(rows: list[dict], model: str, items: dict) -> int:
    changed = 0
    for r in rows:
        if r["label"].split("/", 1)[0] != model or r.get("response") is None:
            continue
        raw = r.get("raw_response", r["response"])
        new = cut(raw)
        ok = score(items[(r["id"], r["lang"])], new)
        if new != r["response"] or ok != r["correct"]:
            changed += 1
        r.update(raw_response=raw, response=new, correct=ok, rescored="cut_at_turn_end")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--file", type=Path, default=OUT)
    a = ap.parse_args()
    out = a.file
    items = {(i.id, i.lang): i for i in load_items()}
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    shutil.copy(out, out.with_suffix(".before_rescore.jsonl"))
    n = rescore(rows, a.model, items)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{a.model}: {n} satir degisti; yedek {out.with_suffix('.before_rescore.jsonl')}")


if __name__ == "__main__":
    main()
