"""Tokenizer maliyeti Turkceye mi ozgu, eklemeli dillerin genel sorunu mu?

Ayni FLORES+ cumleleri: Turkce, Fince, Macarca (eklemeli) ve Almanca, Fransizca (eklemeli degil).
Her tokenizer icin dil/Ingilizce token orani + %95 bootstrap guven araligi.
Bootstrap ESLESTIRILMIS: ayni cumle indeksleri her dilde birlikte yeniden orneklenir
(paralel korpusta cumle zorlugu dillere ortak; ayri orneklemek araligi gereksiz genisletir).

  python -m trbench.tokenizer_tax.fertility     # once: cumle basina sayilar
  python -m trbench.tokenizer_tax.langs         # tablo + grafik
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trbench.tokenizer_tax.fertility import COUNTS
from trbench.tokenizer_tax.plot import INK, INK2, MERGE, SURFACE

OUT = Path("results/tokenizer_langs.csv")
REF = "eng_Latn"
LANG_ORDER = ["tur_Latn", "fin_Latn", "hun_Latn", "deu_Latn", "fra_Latn"]
LANG_NAME = {"tur_Latn": "Turkish", "fin_Latn": "Finnish", "hun_Latn": "Hungarian",
             "deu_Latn": "German", "fra_Latn": "French"}
# Referans paletin ilk 5 kategorik rengi, sabit sirada (validate_palette: PASS; kontrast
# uyarisi -> her dil ayrica farkli isaretle ve lejantla ayirt edilir, tablo CSV'de).
COLOR = dict(zip(LANG_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"], strict=True))
MARKER = dict(zip(LANG_ORDER, ["o", "s", "D", "^", "v"], strict=True))
B, SEED = 2000, 0


def ratios_with_ci(counts: pd.DataFrame, b: int = B, seed: int = SEED) -> pd.DataFrame:
    """counts: model, lang, sent, tokens -> model, lang, ratio, lo, hi."""
    rng = np.random.default_rng(seed)
    rows = []
    for model, g in counts.groupby("model", sort=False):
        wide = g.pivot(index="sent", columns="lang", values="tokens").dropna()
        n = len(wide)
        idx = rng.integers(0, n, size=(b, n))                   # her model ayni sekilde, tekrarlanabilir
        ref = wide[REF].to_numpy()
        ref_boot = ref[idx].sum(axis=1)
        for lang in wide.columns:
            if lang == REF:
                continue
            x = wide[lang].to_numpy()
            boot = x[idx].sum(axis=1) / ref_boot
            rows.append({"model": model, "lang": lang, "ratio": x.sum() / ref.sum(),
                         "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5), "n_sent": n})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Her tokenizer'da Turkce, eklemeli komsularina gore nerede? (araliklar ortusmuyorsa 'fark var')"""
    out = []
    for model, g in df.groupby("model", sort=False):
        r = g.set_index("lang")
        if not {"tur_Latn", "fin_Latn", "hun_Latn"} <= set(r.index):
            continue
        tr = r.loc["tur_Latn"]
        aggl = r.loc[["fin_Latn", "hun_Latn"]]
        out.append({"model": model, "tr": tr.ratio,
                    "tr_vs_fi_hu": "lower" if tr.hi < aggl.lo.min() else
                                   "higher" if tr.lo > aggl.hi.max() else "within"})
    return pd.DataFrame(out)


def headline(df: pd.DataFrame) -> str:
    """Bulguyu verinin kendisinden yaz: ayni tokenizer'i paylasan modeller tek sayilir."""
    s = summary(df)
    s["label"] = s["model"].map(MERGE).fillna(s["model"])
    s = s.drop_duplicates("label")
    low = int((s["tr_vs_fi_hu"] == "lower").sum())
    return (f"Turkish costs fewer tokens than Finnish and Hungarian in {low} of {len(s)} tokenizers"
            if low * 2 > len(s) else
            f"Turkish costs fewer tokens than Finnish and Hungarian in only {low} of {len(s)} tokenizers")


def plot(df: pd.DataFrame, path: Path = Path("docs/tokenizer_langs.png")) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.copy()
    d["label"] = d["model"].map(MERGE).fillna(d["model"])
    d = d.drop_duplicates(["label", "lang"])
    order = (d[d["lang"] == "tur_Latn"].sort_values("ratio")["label"]).tolist()
    y = {lab: i for i, lab in enumerate(order)}
    off = dict(zip(LANG_ORDER, np.linspace(-0.28, 0.28, len(LANG_ORDER)), strict=True))

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for lang in LANG_ORDER:
        s = d[d["lang"] == lang]
        if s.empty:
            continue
        ys = [y[lab] + off[lang] for lab in s["label"]]
        ax.hlines(ys, s["lo"], s["hi"], color=COLOR[lang], lw=2)
        ax.scatter(s["ratio"], ys, s=34, marker=MARKER[lang], color=COLOR[lang],
                   edgecolor=SURFACE, linewidth=1.2, zorder=3, label=LANG_NAME[lang])
    ax.axvline(1.0, color=INK2, lw=1, ls="--")
    ax.set_yticks(range(len(order)), order, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Tokens relative to English for the same sentences (English = 1.0)", color=INK2)
    ax.tick_params(colors=INK2)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    fig.suptitle(headline(df),
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_title("Tokens per sentence relative to English on 2,009 parallel FLORES+ sentences. "
                 "Lower is cheaper; 95% bootstrap intervals are narrower than the markers",
                 loc="left", color=INK2, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    counts = pd.read_csv(COUNTS)
    df = ratios_with_ci(counts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(OUT, index=False)
    wide = df.pivot(index="model", columns="lang", values="ratio")[LANG_ORDER].round(2)
    print(wide.sort_values("tur_Latn").rename(columns=LANG_NAME).to_string())
    print("\nTurkce, Fince/Macarca araliklarina gore:")
    print(summary(df).to_string(index=False))
    print(f"\nkaydedildi: {OUT}, {plot(df)}")


if __name__ == "__main__":
    main()
