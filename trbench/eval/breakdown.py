"""Kategori bazli guven araliklari ve kontaminasyon ayrimi.

  python -m trbench.eval.breakdown                       # GPU, 14 model (results/gpu_quality.jsonl)
  python -m trbench.eval.breakdown --file results/phase4.jsonl --label qwen2.5-7b/fp16

Hazir set (public): Belebele, Global-MMLU, GSM8K -> egitim verisinde olabilir (kontaminasyon riski).
Ozgun (original): talimat, JSON, morfoloji -> bu repoda sablondan uretildi, hicbir yerde yayinlanmadi.
Her hucre: dogruluk [%95 Wilson araligi]; TR-EN farki eslestirilmis bootstrap ile.
Ceviri hatali 8 soru cikarilir (analysis.TRANSLATION_ERRORS).
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

from trbench.eval.analysis import TRANSLATION_ERRORS
from trbench.eval.gpu_quality_report import load

PUBLIC = ("reading", "knowledge", "math")
ORIGINAL = ("instruction", "json")          # morfoloji de ozgun ama yalnizca TR, ayri gosterilir
GROUPS = {"public": PUBLIC, "original": ORIGINAL}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def paired(rows: list[dict]) -> dict[str, dict[str, dict[str, bool]]]:
    """label -> soru id -> {dil: dogru}; ceviri hatalari disarida."""
    out: dict[str, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["id"] not in TRANSLATION_ERRORS:
            out[r["label"]][r["id"]][r["lang"]] = bool(r["correct"])
    return out


def gap_ci(pairs: list[tuple[bool, bool]], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """TR - EN dogruluk farki ve eslestirilmis bootstrap %95 araligi."""
    d = [int(t) - int(e) for t, e in pairs]
    if not d:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), boots[int(0.025 * n)], boots[int(0.975 * n) - 1]


def cell(qs: dict[str, dict[str, bool]], cats: tuple[str, ...]) -> dict:
    both = [(v["tr"], v["en"]) for i, v in qs.items()
            if i.split("/")[0] in cats and "tr" in v and "en" in v]
    tr_k, en_k, n = sum(t for t, _ in both), sum(e for _, e in both), len(both)
    g, lo, hi = gap_ci(both)
    return {"n": n, "tr": tr_k / n if n else math.nan, "tr_ci": wilson(tr_k, n),
            "en": en_k / n if n else math.nan, "en_ci": wilson(en_k, n), "gap": g, "gap_ci": (lo, hi)}


def morph(qs: dict[str, dict[str, bool]]) -> tuple[float, tuple[float, float], int]:
    v = [d["tr"] for i, d in qs.items() if i.startswith("morphology/") and "tr" in d]
    return (sum(v) / len(v) if v else math.nan), wilson(sum(v), len(v)), len(v)


def _pct(x: float) -> str:
    return "–" if math.isnan(x) else f"{100 * x:.0f}"


def _ci(c: tuple[float, float]) -> str:
    return f"[{_pct(c[0])}, {_pct(c[1])}]"


def _gap(c: dict) -> str:
    lo, hi = c["gap_ci"]
    return f"{100 * c['gap']:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}]"


def group_table(data: dict) -> str:
    out = ["| Model | Public TR | Public EN | Public TR−EN | Original TR | Original EN | Original TR−EN |",
           "|---|---|---|---|---|---|---|"]
    rows = []
    for lab, qs in data.items():
        p, o = cell(qs, PUBLIC), cell(qs, ORIGINAL)
        rows.append((-(p["tr"] if not math.isnan(p["tr"]) else 0), lab, p, o))
    for _, lab, p, o in sorted(rows):
        out.append(f"| {lab.split('/')[0]} | {_pct(p['tr'])} {_ci(p['tr_ci'])} | {_pct(p['en'])} {_ci(p['en_ci'])} "
                   f"| {_gap(p)} | {_pct(o['tr'])} {_ci(o['tr_ci'])} | {_pct(o['en'])} {_ci(o['en_ci'])} "
                   f"| {_gap(o)} |")
    return "\n".join(out)


def category_table(data: dict) -> str:
    cats = (*PUBLIC, *ORIGINAL)
    out = ["| Model | " + " | ".join(f"{c} TR−EN" for c in cats) + " | morphology TR |",
           "|---" * (len(cats) + 2) + "|"]
    for lab, qs in sorted(data.items()):
        m, mci, _ = morph(qs)
        out.append(f"| {lab.split('/')[0]} | " + " | ".join(_gap(cell(qs, (c,))) for c in cats)
                   + f" | {_pct(m)} {_ci(mci)} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=Path("results/gpu_quality.jsonl"))
    ap.add_argument("--label", action="append", help="yalnizca bu etiket(ler); varsayilan hepsi")
    a = ap.parse_args()
    data = paired(load(a.file))
    if a.label:
        data = {k: v for k, v in data.items() if k in a.label}
    print("Hazir set (public) ve ozgun maddeler, dogruluk % [%95 Wilson], TR−EN puan [%95 eslestirilmis bootstrap]\n")
    print(group_table(data))
    print("\nKategori bazli TR−EN farki (puan, %95 eslestirilmis bootstrap) ve morfoloji\n")
    print(category_table(data))


if __name__ == "__main__":
    main()
