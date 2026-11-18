"""Kalite x hiz (CPU): her model/quant icin Turkce dogruluk ve Turkce icerik hizi.

Turkce icerik hizi = llama-bench tg128 tok/s / tokenizer vergisi (Faz 1):
ayni Turkce icerigi saniyede ne kadar urettigini, Ingilizce-esdeger token cinsinden verir.

Kullanim:  python -m trbench.eval.pareto   -> tablo + docs/pareto_cpu.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from trbench.eval.analysis import PAIRED, TRANSLATION_ERRORS, load

SPEED = Path("results/cpu_speed.jsonl")
PNG = Path("docs/pareto_cpu.png")
TAX = {"qwen2.5-7b": 1.62, "trendyol-7b": 1.22, "llama-3.1-8b": 1.40, "gemma-2-9b": 1.40,
       "aya-expanse-8b": 1.21, "cosmos-tr-llama-8b": 1.40, "mistral-7b-v0.3": 2.21}
NAME = re.compile(r"(?P<model>.+)-(?P<quant>fp16|q8_0|q4_k_m)\.gguf$")
# Qwen'in resmi GGUF'lari (kendimiz uretmedik): qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf
OFFICIAL = re.compile(r"^qwen2\.5-7b-instruct-(?P<quant>fp16|q8_0|q4_k_m)(-\d{5}-of-\d{5})?\.gguf$")


def speeds(lines: list[str]) -> dict[str, float]:
    """llama-bench jsonl -> {'gemma-2-9b/q8_0': tg tok/s}"""
    out = {}
    for x in lines:
        r = json.loads(x)
        fn = Path(r["model_filename"]).name
        o = OFFICIAL.search(fn)
        m = NAME.search(fn)
        label = f"qwen2.5-7b/{o['quant']}" if o else (f"{m['model']}/{m['quant']}" if m else None)
        if label and r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
            out[label] = float(r["avg_ts"])
    return out


def tr_accuracy(rows: list[dict]) -> dict[str, float]:
    """Etiket basina Turkce dogruluk: eslenik kategoriler, ceviri hatalari haric."""
    hit, n = {}, {}
    for r in rows:
        if r["lang"] != "tr" or r["id"] in TRANSLATION_ERRORS or r["id"].split("/")[0] not in PAIRED:
            continue
        hit[r["label"]] = hit.get(r["label"], 0) + bool(r["correct"])
        n[r["label"]] = n.get(r["label"], 0) + 1
    return {k: hit[k] / n[k] for k in n}


def frontier(points: list[tuple[str, float, float]]) -> list[str]:
    """Hiz ve kalitede baska bir nokta tarafindan gecilmeyenler."""
    return [a for a, x, y in points
            if not any((x2 >= x and y2 >= y) and (x2 > x or y2 > y) for _, x2, y2 in points)]


def main() -> None:
    sp = speeds(SPEED.read_text(encoding="utf-8").splitlines())
    acc = tr_accuracy(load())
    pts = []
    for label in sorted(set(sp) & set(acc)):
        model = label.split("/")[0]
        pts.append((label, sp[label] / TAX[model], 100 * acc[label]))
    best = set(frontier(pts))
    print("| Model / quant | CPU tok/s | TR content tok/s | TR accuracy % | Pareto |")
    print("|---|---|---|---|---|")
    for label, x, y in sorted(pts, key=lambda p: -p[2]):
        print(f"| {label} | {sp[label]:.1f} | {x:.1f} | {y:.0f} | {'yes' if label in best else ''} |")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"gemma-2-9b": "#2a6fdb", "qwen2.5-7b": "#e07b28", "llama-3.1-8b": "#3a9a5b",
              "trendyol-7b": "#8a5cc2", "aya-expanse-8b": "#c2416b", "cosmos-tr-llama-8b": "#1f9aa6",
              "mistral-7b-v0.3": "#8c6d3f"}
    names = {"gemma-2-9b": "Gemma-2-9B", "qwen2.5-7b": "Qwen2.5-7B", "llama-3.1-8b": "Llama-3.1-8B",
             "trendyol-7b": "Trendyol-7B", "aya-expanse-8b": "Aya-Expanse-8B",
             "cosmos-tr-llama-8b": "Cosmos-TR-8B", "mistral-7b-v0.3": "Mistral-7B"}
    marks = {"fp16": "o", "q8_0": "s", "q4_k_m": "^"}
    at = {label: (x, y) for label, x, y in pts}
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    for model, col in colors.items():
        seq = [at[f"{model}/{q}"] for q in marks if f"{model}/{q}" in at]
        if not seq:
            continue
        ax.plot([p[0] for p in seq], [p[1] for p in seq], color=col, lw=1.5, zorder=2)
        for q, m in marks.items():
            if f"{model}/{q}" in at:
                x, y = at[f"{model}/{q}"]
                ax.scatter(x, y, marker=m, s=55, color=col, edgecolor="white", linewidth=0.8, zorder=3)
        x, y = seq[-1]
        ax.annotate(names[model], (x, y), xytext=(8, -3), textcoords="offset points",
                    fontsize=9, color=col, fontweight="bold")
    ax.legend(handles=[Line2D([], [], marker=m, ls="", color="#555", label=q.upper())
                       for q, m in marks.items()],
              loc="lower left", frameon=False, fontsize=8, title="quant", title_fontsize=8)
    ax.set_xlabel("Turkish content throughput on CPU (EN-equivalent tok/s, 16 cores)")
    ax.set_ylabel("Turkish accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(x for _, x, _ in pts) * 1.25)
    ax.grid(color="#eee")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Q4 buys 2–3× the speed for at most 1 point of Turkish accuracy", loc="left", fontsize=11)
    PNG.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(PNG)
    print(f"kaydedildi: {PNG}")


if __name__ == "__main__":
    main()
