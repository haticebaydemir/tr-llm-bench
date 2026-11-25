"""Faz 8c: sonuclar "hazir set" ve "ozgun madde" diye ayri (kontaminasyon kontrolu).

  python -m trbench.eval.contamination

Hazir set (public): Belebele (reading), Global-MMLU CA (knowledge), GSM8K / GSM8K-TR (math).
Bunlar internette yayinli; modelin egitim verisinde olabilirler.
Ozgun (original): talimat (instruction), JSON cikarma (json), morfoloji (yalnizca TR);
bu proje icin sablonla uretildi, once hicbir yerde yayinlanmadi.

Kontaminasyon hazir sette dogrulugu sisirir. Iki sinyal raporlanir:
- her model icin hazir ve ozgun dogruluk (TR, EN) ve TR-EN farki;
- model siralamasi iki kume arasinda tutarli mi (Spearman). Siralama tutarliysa
  sonuc hazir setteki ezberlemeye dayanmiyor.
Bu bir kanit degil gostergedir: iki kume ayni beceriyi olcmuyor.
Ceviri hatali 8 soru cikarilir. Kaynak: results/gpu_quality.jsonl.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from trbench.eval.analysis import TRANSLATION_ERRORS
from trbench.eval.gpu_quality_report import load

GPU = Path("results/gpu_quality.jsonl")
PUBLIC = ("reading", "knowledge", "math")
ORIGINAL = ("instruction", "json")


def split_of(category: str) -> str | None:
    return "public" if category in PUBLIC else "original" if category in ORIGINAL else None


def table(rows: list[dict]) -> pd.DataFrame:
    acc: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        s = split_of(r["category"])
        if s is None or r["id"] in TRANSLATION_ERRORS:
            continue
        k = (r["label"].split("/")[0], s, r["lang"])
        acc[k][0] += int(bool(r["correct"]))
        acc[k][1] += 1
    df = pd.DataFrame([{"model": m, "split": s, "lang": lg, "acc": c / n, "n": n}
                       for (m, s, lg), (c, n) in acc.items()])
    w = df.pivot_table(index="model", columns=["split", "lang"], values="acc")
    w.columns = [f"{s}_{lg}" for s, lg in w.columns]
    w["public_gap"] = w["public_tr"] - w["public_en"]
    w["original_gap"] = w["original_tr"] - w["original_en"]
    w.attrs["n"] = df.groupby(["split", "lang"])["n"].first().to_dict()
    return w.sort_values("public_tr", ascending=False)


def to_md(w: pd.DataFrame) -> str:
    n = w.attrs.get("n", {})
    out = [(f"Items per model: public TR/EN {n.get(('public', 'tr'))}/{n.get(('public', 'en'))}, "
           f"original TR/EN {n.get(('original', 'tr'))}/{n.get(('original', 'en'))}."), "",
           ("| Model | Public TR | Public EN | Public TR−EN | Original TR | Original EN "
           "| Original TR−EN |"),
           "|---|---|---|---|---|---|---|"]
    for m, r in w.iterrows():
        out.append(f"| {m} | {r.public_tr:.0%} | {r.public_en:.0%} | {r.public_gap * 100:+.0f} | "
                   f"{r.original_tr:.0%} | {r.original_en:.0%} | {r.original_gap * 100:+.0f} |")
    rho_tr = w["public_tr"].rank().corr(w["original_tr"].rank())
    rho_en = w["public_en"].rank().corr(w["original_en"].rank())
    out += ["", (f"Rank agreement public vs original (Spearman): TR {rho_tr:+.2f}, EN {rho_en:+.2f} "
            f"(n = {len(w)} models)."),
            (f"Mean TR−EN gap: public {w.public_gap.mean() * 100:+.1f}, "
            f"original {w.original_gap.mean() * 100:+.1f} points.")]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=GPU)
    print(to_md(table(load(ap.parse_args().file))))


if __name__ == "__main__":
    main()
