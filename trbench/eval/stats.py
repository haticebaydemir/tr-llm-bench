"""Faz 8a + 8b: "fark yok" iddialarinin siniri ve modeller arasi eslestirilmis testler.

  python -m trbench.eval.stats                 # ikisi birden, markdown
  python -m trbench.eval.stats --json out.json # ayni sonuclar makine okunur

8a - Dislanabilen en buyuk kayip (KV-cache: results/gpu_kvcache.jsonl,
     agirlik quantization: results/phase4.jsonl varsa).
  Her karsilastirma icin TR kaybi, EN kaybi ve vergi (= TR kaybi - EN kaybi), puan.
  Kayip > 0 = dusuk hassasiyet daha kotu. Soru-ciftleri uzerinden bootstrap.
  - %95 CI: kaybin ust siniri "bundan buyuk kayip dislanabiliyor" degeridir.
  - TOST (%90 CI): esdegerlik marji veriye bakilarak secilmesin diye sabit marj yok;
    yerine esdegerligin saglandigi EN KUCUK marj raporlanir = max(|alt90|, |ust90|).
    Bu marj ancak gercekten anlamliysa (ornegin 2-3 puan) "esdeger" denebilir.

8b - Modeller arasi (results/gpu_quality.jsonl, 14 model, ayni sorular).
  Her dil icin tum ikili karsilastirmalar: exact McNemar (uyusmayan ciftler uzerinden
  binom testi), Holm duzeltmesi. Kademe: en iyi modelle anlamli farki olmayanlar 1. kademe;
  kalanlarin en iyisiyle anlamli farki olmayanlar 2. kademe, ...
  Ceviri hatali 8 soru cikarilir; morfoloji yalnizca TR, burada dahil degil.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from trbench.eval.analysis import BASE, PAIRED, TRANSLATION_ERRORS, cpu_results, paired_ids
from trbench.eval.analysis import load as load_cpu
from trbench.eval.analysis import table as cpu_table
from trbench.eval.gpu_quality_report import load
from trbench.eval.kv_report import tables as kv_tables

KV = Path("results/gpu_kvcache.jsonl")
CPU = cpu_results()
GPU = Path("results/gpu_quality.jsonl")
N_BOOT = 4000
ALPHA = 0.05


# ---------- 8a ----------

def _losses(b: dict, q: dict, ids: list[str]) -> tuple[float, float, float]:
    n = len(ids)
    ltr = sum(b[(i, "tr")] - q[(i, "tr")] for i in ids) / n
    len_ = sum(b[(i, "en")] - q[(i, "en")] for i in ids) / n
    return ltr, len_, ltr - len_


def _q(vals: list[float], p: float) -> float:
    """Basit yuzdelik (kv_report/analysis ile ayni kural)."""
    k = min(len(vals) - 1, max(0, int(p * len(vals)) - (1 if p > 0.5 else 0)))
    return vals[k]


def equivalence(tab: dict, quant: str, n: int = N_BOOT, seed: int = 0) -> dict:
    ids = [i for i in paired_ids(tab, quant) if i not in TRANSLATION_ERRORS]
    b, q = tab[BASE], tab[quant]
    point = _losses(b, q, ids)
    rng = random.Random(seed)
    boot = [_losses(b, q, [rng.choice(ids) for _ in ids]) for _ in range(n)]
    out = {"n_pairs": len(ids)}
    for k, name in enumerate(("loss_tr", "loss_en", "tax")):
        v = sorted(x[k] * 100 for x in boot)
        lo95, hi95 = _q(v, 0.025), _q(v, 0.975)
        lo90, hi90 = _q(v, 0.05), _q(v, 0.95)
        out[name] = {"est": point[k] * 100, "ci95": [lo95, hi95], "ci90": [lo90, hi90],
                     "max_excluded_loss": hi95,
                     "min_equiv_margin": max(abs(lo90), abs(hi90))}
    return out


def equivalence_rows() -> list[dict]:
    rows = []
    if KV.exists():
        for model, tab in sorted(kv_tables(load(KV)).items()):
            for q in ("q8_0", "q4_0"):
                if BASE in tab and q in tab:
                    rows.append({"kind": "kv-cache", "model": model, "vs": f"f16 -> {q}",
                                 **equivalence(tab, q)})
    if CPU.exists():
        crow = load_cpu(CPU)
        for model in sorted({r["label"].split("/", 1)[0] for r in crow}):
            tab = cpu_table(crow, model)
            for q in ("q8_0", "q4_k_m"):
                if BASE in tab and q in tab:
                    rows.append({"kind": "weights", "model": model, "vs": f"fp16 -> {q}",
                                 **equivalence(tab, q)})
    return rows


def equivalence_md(rows: list[dict]) -> str:
    f = "{:+.1f}".format
    out = [("| Type | Model | Comparison | Pairs | TR loss [95% CI] | Largest TR loss excluded "
            "| Tax (TR-EN) [95% CI] | Smallest equivalence margin for tax (TOST, 90%) |"),
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lt, t = r["loss_tr"], r["tax"]
        out.append(f"| {r['kind']} | {r['model']} | {r['vs']} | {r['n_pairs']} | "
                   f"{f(lt['est'])} [{f(lt['ci95'][0])}, {f(lt['ci95'][1])}] | "
                   f"{lt['max_excluded_loss']:.1f} | "
                   f"{f(t['est'])} [{f(t['ci95'][0])}, {f(t['ci95'][1])}] | "
                   f"±{t['min_equiv_margin']:.1f} |")
    return "\n".join(out)


# ---------- 8b ----------

def mcnemar_exact(b: int, c: int) -> float:
    """Iki tarafli exact McNemar: uyusmayan n=b+c cift, H0: p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni duzeltilmis p degerleri (monoton)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for rank, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - rank) * p))
        out[k] = run
    return out


def answers(rows: list[dict], lang: str) -> dict[str, dict[str, bool]]:
    """model -> {soru_id: dogru}; yalnizca paralel kategoriler, ceviri hatalari haric."""
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    for r in rows:
        if (r["lang"] == lang and r["category"] in PAIRED
                and r["id"] not in TRANSLATION_ERRORS):
            out[r["label"].split("/")[0]][r["id"]] = bool(r["correct"])
    return out


def pairwise(ans: dict[str, dict[str, bool]]) -> dict:
    models = sorted(ans)
    common = sorted(set.intersection(*(set(ans[m]) for m in models)))
    acc = {m: sum(ans[m][i] for i in common) / len(common) for m in models}
    raw, disc = {}, {}
    for a, b in combinations(models, 2):
        nb = sum(ans[a][i] and not ans[b][i] for i in common)
        nc = sum(ans[b][i] and not ans[a][i] for i in common)
        raw[(a, b)], disc[(a, b)] = mcnemar_exact(nb, nc), (nb, nc)
    adj = holm(raw)
    sig = {frozenset(k) for k, p in adj.items() if p < ALPHA}
    order = sorted(models, key=lambda m: -acc[m])
    tier, left, t = {}, list(order), 0
    while left:
        t += 1
        top = left[0]
        grp = [m for m in left if m == top or frozenset((top, m)) not in sig]
        for m in grp:
            tier[m] = t
        left = [m for m in left if m not in grp]
    return {"n_items": len(common), "acc": acc, "tier": tier, "order": order,
            "pairs": [{"a": a, "b": b, "a_only": disc[(a, b)][0], "b_only": disc[(a, b)][1],
                       "p": raw[(a, b)], "p_holm": adj[(a, b)]} for a, b in raw]}


def pairwise_md(res: dict, lang: str) -> str:
    out = [(f"**{lang.upper()}** — {res['n_items']} items, {len(res['pairs'])} pairs, "
            f"exact McNemar + Holm (α = {ALPHA})"), "",
           "| Tier | Model | Accuracy | Not significantly different from |", "|---|---|---|---|"]
    sig = {frozenset((p["a"], p["b"])) for p in res["pairs"] if p["p_holm"] < ALPHA}
    for m in res["order"]:
        same = [x for x in res["order"] if x != m and frozenset((m, x)) not in sig]
        out.append(f"| {res['tier'][m]} | {m} | {res['acc'][m]:.1%} | {', '.join(same) or '–'} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, help="sonuclari JSON olarak da yaz")
    a = ap.parse_args()

    eq = equivalence_rows()
    print("## 8a — Largest loss that can be excluded\n")
    if not CPU.exists():
        print(f"(not: {CPU} yok, yalnizca KV-cache karsilastirmalari)\n")
    print(equivalence_md(eq))

    rows = load(GPU)
    pw = {lang: pairwise(answers(rows, lang)) for lang in ("tr", "en")}
    print("\n## 8b — Paired model comparisons\n")
    for lang in ("tr", "en"):
        print(pairwise_md(pw[lang], lang) + "\n")

    if a.json:
        a.json.write_text(json.dumps({"equivalence": eq, "pairwise": pw}, indent=1,
                                     ensure_ascii=False), encoding="utf-8")
        print(f"JSON: {a.json}")


if __name__ == "__main__":
    main()
