"""Faz 4 analizi: quantization vergisi.

  kayip_dil(q)  = dogruluk(FP16, dil) - dogruluk(q, dil)
  vergi(q)      = kayip_TR(q) - kayip_EN(q)
  vergi > 0  ->  quantization Turkce'yi Ingilizce'den DAHA COK bozuyor.

Sadece TR/EN paralel kategoriler kiyaslanir (morfoloji yalnizca TR, ayri raporlanir).
Belirsizlik: soru-ciftleri uzerinden bootstrap -> %95 guven araligi.
Aralik 0'i iceriyorsa "fark yok" diyemeyiz ama "fark var" da diyemeyiz.

Kullanim:  python -m trbench.eval.analysis --model qwen2.5-7b
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

RESULTS = Path("results/phase4.jsonl")
PAIRED = ("reading", "knowledge", "math", "instruction", "json")
BASE = "fp16"

# 17 Kasim elle inceleme: Turkce ceviri sorunun anlamini degistiriyor (bkz. README).
TRANSLATION_ERRORS = frozenset({
    "reading/https://en.wikinews.org/wiki/Building_collapses_in_Mecca_at_eve_of_Hajj_pillgrimage#1",
    "reading/https://en.wikinews.org/wiki/Sun%27s_mood_swings_not_so_strange_after_all,_say_scientists#1",
    "reading/https://en.wikivoyage.org/wiki/Aggregators#1",
    "reading/https://en.wikivoyage.org/wiki/Noum%C3%A9a#2",
    "reading/https://en.wikinews.org/wiki/The_Onion:_An_interview_with_%27America%27s_Finest_News_Source%27#2",
    "knowledge/high_school_biology/test/211",
    "knowledge/college_medicine/test/72",
    "math/gsm8k-test-721",
})


def load(path: Path = RESULTS) -> list[dict]:
    from trbench.results import pick
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return pick([r for r in rows if r.get("error") is None], ("label",), ("id", "lang"))


def table(rows: list[dict], model: str) -> dict[str, dict[tuple[str, str], bool]]:
    """quant -> {(soru_id, dil): dogru_mu} (tek model icin)."""
    out: dict[str, dict[tuple[str, str], bool]] = defaultdict(dict)
    for r in rows:
        name, quant = r["label"].split("/", 1)
        if name == model:
            out[quant][(r["id"], r["lang"])] = bool(r["correct"])
    return out


def _acc(t: dict[tuple[str, str], bool], ids: list[str], lang: str) -> float:
    return sum(t[(i, lang)] for i in ids) / len(ids)


def paired_ids(tab: dict, quant: str, cats: tuple[str, ...] = PAIRED) -> list[str]:
    """FP16 ve quant'ta, TR ve EN'de HEPSI olculmus soru id'leri (adil kiyas)."""
    need = [tab[BASE], tab[quant]]
    ids = {i for (i, lang) in tab[BASE] if lang == "tr" and i.split("/")[0] in cats}
    return sorted(i for i in ids if all((i, lg) in t for t in need for lg in ("tr", "en")))


def tax(tab: dict, quant: str, ids: list[str]) -> dict[str, float]:
    b, q = tab[BASE], tab[quant]
    loss_tr = _acc(b, ids, "tr") - _acc(q, ids, "tr")
    loss_en = _acc(b, ids, "en") - _acc(q, ids, "en")
    return {"loss_tr": loss_tr, "loss_en": loss_en, "tax": loss_tr - loss_en}


def bootstrap_ci(tab: dict, quant: str, ids: list[str], n: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """Soru-ciftlerini yerine koyarak yeniden orneklem -> verginin %95 araligi."""
    rng = random.Random(seed)
    vals = sorted(tax(tab, quant, [rng.choice(ids) for _ in ids])["tax"] for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def tr_wrong_en_right(rows: list[dict], label: str) -> list[str]:
    """Ayni soru: TR yanlis, EN dogru -> ceviri hatasi adaylari (elle incelenecek)."""
    by: dict[str, dict[str, bool]] = defaultdict(dict)
    for r in rows:
        if r["label"] == label:
            by[r["id"]][r["lang"]] = bool(r["correct"])
    return sorted(i for i, d in by.items() if d.get("tr") is False and d.get("en") is True)


def models(rows: list[dict]) -> list[str]:
    return sorted({r["label"].split("/", 1)[0] for r in rows})


def gap(t: dict, cat: str, exclude: frozenset = frozenset()) -> tuple[float, float] | None:
    """(TR dogruluk, EN dogruluk) tek kategori; iki dilde de olculen sorular."""
    ids = sorted({i for (i, lg) in t if lg == "tr" and i.startswith(cat + "/")
                  and (i, "en") in t and i not in exclude})
    if not ids:
        return None
    return _acc(t, ids, "tr"), _acc(t, ids, "en")


def summary_md(rows: list[dict], clean: bool = False) -> str:
    """Butun modeller tek tabloda (README icin markdown)."""
    ex = TRANSLATION_ERRORS if clean else frozenset()
    out = ["| Model | " + " | ".join(f"{c} TR/EN" for c in PAIRED)
           + " | Morph. TR | Q8 tax [95% CI] | Q4 tax [95% CI] |",
           "|---" * (len(PAIRED) + 4) + "|"]
    for m in models(rows):
        tab = table(rows, m)
        if BASE not in tab:
            continue
        cells = []
        for c in PAIRED:
            g = gap(tab[BASE], c, ex)
            cells.append("–" if g is None else f"{100*g[0]:.0f} / {100*g[1]:.0f}")
        mo = [v for (i, _), v in tab[BASE].items() if i.startswith("morphology/")]
        cells.append(f"{100*sum(mo)/len(mo):.0f}" if mo else "–")
        for q in ("q8_0", "q4_k_m"):
            ids = [i for i in paired_ids(tab, q) if i not in ex] if q in tab else []
            if not ids:
                cells.append("–")
                continue
            t = tax(tab, q, ids)
            lo, hi = bootstrap_ci(tab, q, ids)
            cells.append(f"{100*t['tax']:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--all", action="store_true", help="butun modeller, markdown tablo")
    ap.add_argument("--clean", action="store_true", help="ceviri hatali 8 soruyu cikar")
    a = ap.parse_args()
    rows = load()
    if a.all:
        print(f"fp16 dogruluk (%) ve quantization vergisi (puan){' - ceviri hatalari cikarildi' if a.clean else ''}")
        print(summary_md(rows, a.clean))
        return
    tab = table(rows, a.model)
    if BASE not in tab:
        raise SystemExit(f"{a.model}/{BASE} henuz yok -> vergi hesaplanamaz. Mevcut: {sorted(tab)}")

    print(f"== {a.model}: quantization vergisi (puan; vergi>0 = TR daha cok kaybediyor) ==")
    for q in sorted(k for k in tab if k != BASE):
        ids = paired_ids(tab, q)
        if not ids:
            continue
        t = tax(tab, q, ids)
        lo, hi = bootstrap_ci(tab, q, ids)
        print(f"  {q:8s} n={len(ids):3d}  kayip TR {100*t['loss_tr']:+5.1f}  "
              f"EN {100*t['loss_en']:+5.1f}  vergi {100*t['tax']:+5.1f}  "
              f"%95 GA [{100*lo:+5.1f}, {100*hi:+5.1f}]")
        for c in PAIRED:
            cids = [i for i in ids if i.startswith(c + "/")]
            if cids:
                ct = tax(tab, q, cids)
                print(f"      {c:12s} n={len(cids):3d}  TR {100*ct['loss_tr']:+5.1f}  "
                      f"EN {100*ct['loss_en']:+5.1f}  vergi {100*ct['tax']:+5.1f}")

    print("\n== Morfoloji (sadece TR) ==")
    for q, t in sorted(tab.items()):
        m = [v for (i, lang), v in t.items() if i.startswith("morphology/")]
        if m:
            print(f"  {q:8s} {sum(m)}/{len(m)}  %{100 * sum(m) / len(m):.1f}")

    lab = f"{a.model}/{BASE}"
    cand = tr_wrong_en_right(rows, lab)
    print(f"\n== {lab}: TR yanlis / EN dogru: {len(cand)} soru (ceviri hatasi adaylari) ==")
    for i in cand[:15]:
        print("  ", i)


if __name__ == "__main__":
    main()
