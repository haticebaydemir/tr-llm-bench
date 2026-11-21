"""KV-cache quantization vergisi (results/gpu_kvcache.jsonl).

  python -m trbench.eval.kv_report

Ayni model, ayni Q4_K_M agirliklar; sadece cache tipi degisir (f16 / q8_0 / q4_0, -fa on).
vergi = (TR kaybi) - (EN kaybi), f16 cache'e gore; soru-ciftleri uzerinden %95 bootstrap.
Ceviri hatalari cikarilir; morfoloji yalnizca TR, ayri gosterilir.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from trbench.eval.analysis import BASE, TRANSLATION_ERRORS, bootstrap_ci, paired_ids, tax
from trbench.eval.gpu_quality_report import load

KV = Path("results/gpu_kvcache.jsonl")
CACHES = {"kvf16": BASE, "kvq8": "q8_0", "kvq4": "q4_0"}


def tables(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """model -> cache -> {(id, dil): dogru}; cache f16 analysis.BASE adini alir."""
    out: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        model, _, suf = r["label"].split("/")[0].rpartition("-")
        if suf in CACHES and r["id"] not in TRANSLATION_ERRORS:
            out[model][CACHES[suf]][(r["id"], r["lang"])] = bool(r["correct"])
    return out


def morph(t: dict) -> float:
    v = [c for (i, lg), c in t.items() if i.startswith("morphology/")]
    return sum(v) / len(v) if v else float("nan")


def report(rows: list[dict]) -> str:
    out = ["| Model | KV cache | TR loss | EN loss | KV tax [95% CI] | Morphology (TR) |",
           "|---|---|---|---|---|---|"]
    for m, tab in sorted(tables(rows).items()):
        if BASE not in tab:
            continue
        out.append(f"| {m} | f16 | – | – | – | {morph(tab[BASE]):.0%} |")
        for q in ("q8_0", "q4_0"):
            if q not in tab:
                continue
            ids = paired_ids(tab, q)
            t = tax(tab, q, ids)
            lo, hi = bootstrap_ci(tab, q, ids)
            out.append(f"| {m} | {q} | {t['loss_tr']*100:+.1f} | {t['loss_en']*100:+.1f} | "
                       f"{t['tax']*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}] | {morph(tab[q]):.0%} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=KV)
    a = ap.parse_args()
    rows = load(a.file)
    print(f"satir: {len(rows)}\n")
    print(report(rows))


if __name__ == "__main__":
    main()
