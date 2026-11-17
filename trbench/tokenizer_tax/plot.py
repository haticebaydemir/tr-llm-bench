"""Manset grafigi: ayni Turkce metin, tokenizer'a gore kac kat token tutuyor?

Kullanim:
  python -m trbench.tokenizer_tax.plot            # Ingilizce
  python -m trbench.tokenizer_tax.plot --lang tr  # Turkce
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trbench.tokenizer_tax.fertility import OUT as CSV

# Ayni tokenizer'i paylasan modeller tek cubukta: tekrar bilgi tasimaz.
MERGE = {
    "Qwen2.5-7B": "Qwen2.5 / Qwen3",
    "Qwen3-8B": "Qwen2.5 / Qwen3",
    "Llama-3.1-8B": "Llama-3.1 (= Cosmos-TR)",
    "Cosmos-TR-Llama-8B": "Llama-3.1 (= Cosmos-TR)",
}

TEXT = {
    "en": {
        "title": "The same Turkish text takes 1.2× to 2.2× as many tokens as English",
        "sub": "Turkish / English token count on 2,009 parallel FLORES+ sentences, by tokenizer",
        "x": "Tokens relative to English (English = 1.0)",
        "note": "Lower is cheaper and faster. Dashed line = English. Source: TR-LLM Bench",
    },
    "tr": {
        "title": "Aynı Türkçe metin, İngilizcenin 1.2 ile 2.2 katı token tutuyor",
        "sub": "2.009 paralel FLORES+ cümlesinde Türkçe / İngilizce token oranı, tokenizer'a göre",
        "x": "İngilizceye göre token (İngilizce = 1.0)",
        "note": "Düşük olan daha ucuz ve hızlı. Kesikli çizgi = İngilizce. Kaynak: TR-LLM Bench",
    },
}

# Referans paletten (dataviz skill): tek seri -> tek renk, metin mürekkep tonlarında
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def tr_ratios(df: pd.DataFrame) -> pd.Series:
    tr = df[df["lang"] == "tur_Latn"].copy()
    tr["label"] = tr["model"].map(MERGE).fillna(tr["model"])
    return tr.groupby("label")["tax_vs_eng"].first().sort_values()


def plot(lang: str = "en", out_dir: Path = Path("results")) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = TEXT[lang]
    s = tr_ratios(pd.read_csv(CSV))

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y = range(len(s))
    # Cubuklar sifirdan baslar: 1.0'dan baslatmak farklari abartirdi.
    ax.barh(list(y), s.values, height=0.62, color=SERIES, edgecolor=SURFACE)
    ax.axvline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    for i, v in zip(y, s.values):
        ax.text(v + 0.015, i, f"{v:.2f}×", va="center", fontsize=10, color=INK)

    ax.set_yticks(list(y), s.index, fontsize=10, color=INK)
    ax.set_xlim(0, max(s.values) + 0.25)
    ax.set_xlabel(t["x"], fontsize=10, color=INK2)
    ax.tick_params(axis="x", colors=INK2, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.text(0.02, 0.965, t["title"], fontsize=14, fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.925, t["sub"], fontsize=10, color=INK2, ha="left")
    fig.text(0.02, 0.02, t["note"], fontsize=8.5, color=INK2, ha="left")
    fig.subplots_adjust(left=0.24, right=0.96, top=0.86, bottom=0.13)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tokenizer_tax_{lang}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("kaydedildi:", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "tr"], default="en")
    plot(ap.parse_args().lang)


if __name__ == "__main__":
    main()
