"""Faz 8f: tokenizer vergisinin dagilimi (mevcut veriden, yeni kosu yok).

  python -m trbench.tokenizer_tax.distribution            # markdown, stdout
  python -m trbench.tokenizer_tax.distribution --json out.json

Girdi: results/tokenizer_counts.csv.gz (cumle basina token sayisi, metin yok) ve
results/tokenizer_tax.csv (vocab boyutu). Ana iddia "Turkce her tokenizer'da pahali" degil,
"maliyet tokenizer'a gore 0.64x-2.21x degisiyor"; bu script o degiskenligi ayristirir:

1. Ozdes tokenizer'lar: tum dillerde cumle cumle ayni sayiyi verenler tek satira iner
   (orn. Llama-3.1-8B = Cosmos-TR-Llama-8B). Bagimsiz tokenizer sayisi bundan gelir.
2. Cumle duzeyi dagilim: TR/EN oraninin medyani, p10-p90 ve Turkcenin Ingilizceden ucuz
   oldugu cumle orani. Toplam oran (README) ile medyan farkliysa uzun cumleler agirlik yapiyor.
3. Cumle uzunluguna gore: Ingilizce kelime yerine ortak bir olcu gerekiyor; her cumlenin
   uzunlugu = 17 tokenizer'in Ingilizce token sayisinin medyani. Bes esit dilim (quintile),
   her dilimde toplam TR/EN orani.
4. Turkceye ozgu maliyet: TR / ortalama(FI, HU). < 1 ise Turkce, eklemeli komsularindan ucuz;
   tokenizer'in "gordugu Turkce miktari" tezinin dogrudan gostergesi. Vocab boyutuyla
   Spearman korelasyonu (bagimsiz tokenizer'lar uzerinden).
5. Metin kaynagi: FLORES+ dev (ilk 997 cumle) ve devtest (sonraki 1012) ayri ayri oran.
   Iki alt kume ayni sonucu veriyorsa oran tek bir metin parcasina bagli degil.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COUNTS = Path("results/tokenizer_counts.csv.gz")
TAX = Path("results/tokenizer_tax.csv")
EN, TR, FI, HU = "eng_Latn", "tur_Latn", "fin_Latn", "hun_Latn"
DEV_N = 997  # FLORES+ dev cumle sayisi; corpus.align id'leri dev_ < devtest_ diye siralar
N_BINS = 5


def wide(counts: pd.DataFrame) -> pd.DataFrame:
    """index (model, sent), sutun = dil."""
    return counts.pivot_table(index=["model", "sent"], columns="lang", values="tokens").dropna()


def identical_groups(w: pd.DataFrame) -> list[list[str]]:
    """Tum dillerde cumle cumle ayni sayiyi veren modelleri grupla (ilk gorulen temsilci)."""
    sig: dict[bytes, list[str]] = {}
    for model, g in w.groupby(level="model", sort=False):
        key = np.ascontiguousarray(g.sort_index(axis=1).to_numpy(dtype=np.int64)).tobytes()
        sig.setdefault(key, []).append(model)
    return list(sig.values())


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = pd.Series(x).rank(), pd.Series(y).rank()
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse(counts: pd.DataFrame, vocab: dict[str, int]) -> dict:
    w = wide(counts)
    groups = identical_groups(w)
    reps = [g[0] for g in groups]
    names = {g[0]: " / ".join(g) for g in groups}

    # ortak uzunluk olcusu: Ingilizce token sayisinin tokenizer'lar arasi medyani
    length = w[EN].groupby(level="sent").median()
    bins = pd.qcut(length, N_BINS, labels=False, duplicates="drop")

    rows = []
    for m in reps:
        g = w.xs(m, level="model")
        tr, en = g[TR], g[EN]
        r = tr / en
        dev, test = g.index < DEV_N, g.index >= DEV_N
        by_len = [float(tr[bins == b].sum() / en[bins == b].sum()) for b in range(int(bins.max()) + 1)]
        rows.append({
            "tokenizer": names[m], "vocab": vocab.get(m),
            "tax": float(tr.sum() / en.sum()),
            "median": float(r.median()), "p10": float(r.quantile(.1)), "p90": float(r.quantile(.9)),
            "tr_cheaper_share": float((tr < en).mean()),
            "by_length": by_len,
            "tr_vs_fi_hu": float(tr.sum() / ((g[FI].sum() + g[HU].sum()) / 2)),
            "dev": float(tr[dev].sum() / en[dev].sum()),
            "devtest": float(tr[test].sum() / en[test].sum()),
        })
    rows.sort(key=lambda r: r["tax"])
    v = [r for r in rows if r["vocab"]]
    edges = [float(length[bins == b].min()) for b in range(int(bins.max()) + 1)] + [float(length.max())]
    return {
        "n_tokenizers": len(counts["model"].unique()), "n_independent": len(reps),
        "n_sent": len(length), "length_edges": edges, "rows": rows,
        "spearman_vocab_tax": spearman([r["vocab"] for r in v], [r["tax"] for r in v]),
        "spearman_vocab_tr_vs_fi_hu": spearman([r["vocab"] for r in v], [r["tr_vs_fi_hu"] for r in v]),
        "max_split_gap": max(abs(r["dev"] - r["devtest"]) for r in rows),
    }


def to_md(a: dict) -> str:
    e = a["length_edges"]
    bins = [f"Q{i + 1} ({e[i]:.0f}–{e[i + 1]:.0f})" for i in range(len(e) - 1)]
    out = [(f"{a['n_tokenizers']} tokenizers, {a['n_independent']} independent "
           f"(identical per-sentence counts merged), {a['n_sent']} FLORES+ sentences."), "",
           "### Sentence-level distribution and Turkish-specific cost", "",
           ("| Tokenizer | Vocab | Tax (total) | Median | p10–p90 | TR cheaper than EN "
           "| TR / mean(FI, HU) | dev | devtest |"),
           "|---|---|---|---|---|---|---|---|---|"]
    for r in a["rows"]:
        out.append(f"| {r['tokenizer']} | {r['vocab'] or '–'} | {r['tax']:.2f} | {r['median']:.2f} | "
                   f"{r['p10']:.2f}–{r['p90']:.2f} | {r['tr_cheaper_share']:.0%} | "
                   f"{r['tr_vs_fi_hu']:.2f} | {r['dev']:.2f} | {r['devtest']:.2f} |")
    out += ["", "### Tax by sentence length (EN tokens, median across tokenizers)", "",
            "| Tokenizer | " + " | ".join(bins) + " |", "|---" * (len(bins) + 1) + "|"]
    for r in a["rows"]:
        out.append(f"| {r['tokenizer']} | " + " | ".join(f"{x:.2f}" for x in r["by_length"]) + " |")
    out += ["", (f"Spearman(vocab, tax) = {a['spearman_vocab_tax']:+.2f}; "
            f"Spearman(vocab, TR / mean(FI, HU)) = {a['spearman_vocab_tr_vs_fi_hu']:+.2f} "
            f"(n = {a['n_independent']})."),
            f"Largest dev/devtest gap in the tax: {a['max_split_gap']:.3f}."]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    counts = pd.read_csv(COUNTS)
    vocab = pd.read_csv(TAX).drop_duplicates("model").set_index("model")["vocab_size"].to_dict()
    res = analyse(counts, vocab)
    print(to_md(res))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nJSON: {a.json}")


if __name__ == "__main__":
    main()
