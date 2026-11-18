"""Yayin ciktilari: quantization vergisi grafigi + statik leaderboard (HF Space).

Kullanim:  python -m trbench.eval.report
  -> docs/quant_tax.png
  -> space/index.html, space/README.md   (hf upload HaticeB/tr-llm-bench space . --repo-type space)
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from trbench.eval.analysis import (
    BASE,
    PAIRED,
    TRANSLATION_ERRORS,
    bootstrap_ci,
    load,
    models,
    paired_ids,
    table,
    tax,
)
from trbench.eval.pareto import SPEED, TAX, speeds, tr_accuracy

PNG = Path("docs/quant_tax.png")
SPACE = Path("space")
QUANTS = ("fp16", "q8_0", "q4_k_m")


def tax_rows(rows: list[dict]) -> list[dict]:
    """Model x quant: vergi + %95 GA (ceviri hatalari haric)."""
    out = []
    for m in models(rows):
        tab = table(rows, m)
        if BASE not in tab:
            continue
        for q in ("q8_0", "q4_k_m"):
            if q not in tab:
                continue
            ids = [i for i in paired_ids(tab, q) if i not in TRANSLATION_ERRORS]
            t = tax(tab, q, ids)
            lo, hi = bootstrap_ci(tab, q, ids)
            out.append({"model": m, "quant": q, "tax": 100 * t["tax"], "lo": 100 * lo, "hi": 100 * hi})
    return out


def plot_tax(tr: list[dict]) -> None:
    import matplotlib.pyplot as plt
    ms = sorted({r["model"] for r in tr})
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(ms) + 1.6), dpi=150)
    for i, m in enumerate(ms):
        for r in tr:
            if r["model"] != m:
                continue
            y = i + (0.15 if r["quant"] == "q8_0" else -0.15)
            col = "#9aa0a6" if r["quant"] == "q8_0" else "#2a6fdb"
            ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=2)
            ax.scatter(r["tax"], y, color=col, s=30, zorder=3)
    ax.axvline(0, color="#444", lw=1)
    ax.set_yticks(range(len(ms)), ms)
    ax.set_xlabel("Quantization tax: Turkish loss − English loss vs fp16 (points, 95% CI)")
    ax.scatter([], [], color="#9aa0a6", label="Q8_0")
    ax.scatter([], [], color="#2a6fdb", label="Q4_K_M")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#eee")
    ax.set_title("Every interval crosses zero: no measurable quantization tax on Turkish",
                 loc="left", fontsize=11)
    PNG.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(PNG)


def board(rows: list[dict], sp: dict[str, float]) -> list[dict]:
    acc = tr_accuracy(rows)
    out = []
    for m in models(rows):
        tab = table(rows, m)
        for q in QUANTS:
            if q not in tab:
                continue
            t = tab[q]
            en = [v for (i, lg), v in t.items() if lg == "en" and i not in TRANSLATION_ERRORS
                  and i.split("/")[0] in PAIRED]
            mo = [v for (i, lg), v in t.items() if i.startswith("morphology/")]
            label = f"{m}/{q}"
            spd = sp.get(label)
            out.append({
                "model": m, "quant": q,
                "tr": round(100 * acc[label], 1) if label in acc else None,
                "en": round(100 * sum(en) / len(en), 1) if en else None,
                "morph": round(100 * sum(mo) / len(mo), 1) if mo else None,
                "tok_tax": TAX.get(m),
                "cpu_tps": round(spd, 1) if spd else None,
                "tr_content_tps": round(spd / TAX[m], 1) if spd and m in TAX else None,
            })
    return out


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TR-LLM Bench leaderboard</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1d1d1f}}
h1{{font-size:1.5rem;margin:0 0 .25rem}} p{{color:#555;margin:.25rem 0 1rem}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:.45rem .6rem;border-bottom:1px solid #e5e5e5;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
th{{cursor:pointer;user-select:none;background:#fafafa;font-weight:600}} th:hover{{background:#f0f0f0}}
tr:hover td{{background:#f7f9fc}} a{{color:#2a6fdb}} small{{color:#777}}
</style></head><body>
<h1>TR-LLM Bench leaderboard</h1>
<p>Turkish vs English accuracy on 242 paired items (8 mistranslated items removed), Turkish morphology,
and CPU speed (llama.cpp <code>b12470</code>, 16 cores). <b>TR content tok/s</b> = CPU tok/s ÷ tokenizer tax.
Click a column to sort. <a target="_blank" rel="noopener" href="https://github.com/haticebaydemir/tr-llm-bench">Code &amp; method</a> ·
<a target="_blank" rel="noopener" href="https://huggingface.co/datasets/HaticeB/tr-llm-bench-eval">Eval set</a></p>
<table id="t"><thead><tr><th>Model</th><th>Quant</th><th>TR acc %</th><th>EN acc %</th><th>TR−EN</th>
<th>Morph. %</th><th>Tokenizer tax</th><th>CPU tok/s</th><th>TR content tok/s</th></tr></thead><tbody>
{rows}
</tbody></table>
<p><small>Updated {date}. Greedy decoding, deterministic scoring, no LLM judge.</small></p>
<script>
const t=document.getElementById('t');
t.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{{
 const rs=[...t.tBodies[0].rows],num=i>1,asc=th.dataset.asc!=='1';
 rs.sort((a,b)=>{{let x=a.cells[i].dataset.v??a.cells[i].textContent,y=b.cells[i].dataset.v??b.cells[i].textContent;
  if(num){{x=parseFloat(x);y=parseFloat(y);x=isNaN(x)?-1e9:x;y=isNaN(y)?-1e9:y;return asc?x-y:y-x}}
  return asc?x.localeCompare(y):y.localeCompare(x)}});
 th.dataset.asc=asc?'1':'0';rs.forEach(r=>t.tBodies[0].appendChild(r))}});
</script></body></html>
"""

CARD = """---
title: TR-LLM Bench
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: cc-by-4.0
---
Leaderboard for [TR-LLM Bench](https://github.com/haticebaydemir/tr-llm-bench): Turkish vs English
quality and CPU speed of open LLMs across quantization levels.
"""


def cell(v, fmt="{:.1f}") -> str:
    return "<td>–</td>" if v is None else f'<td data-v="{v}">{fmt.format(v)}</td>'


def write_space(b: list[dict], date: str) -> None:
    trs = []
    for r in sorted(b, key=lambda r: -(r["tr"] or 0)):
        gap = None if r["tr"] is None or r["en"] is None else round(r["tr"] - r["en"], 1)
        trs.append("<tr><td>" + html.escape(r["model"]) + "</td><td>" + r["quant"].upper() + "</td>"
                   + cell(r["tr"]) + cell(r["en"]) + cell(gap, "{:+.1f}") + cell(r["morph"])
                   + cell(r["tok_tax"], "{:.2f}×") + cell(r["cpu_tps"]) + cell(r["tr_content_tps"]) + "</tr>")
    SPACE.mkdir(exist_ok=True)
    (SPACE / "index.html").write_text(PAGE.format(rows="\n".join(trs), date=date), encoding="utf-8")
    (SPACE / "README.md").write_text(CARD, encoding="utf-8")
    (SPACE / "data.json").write_text(json.dumps(b, indent=1), encoding="utf-8")


def main() -> None:
    import datetime
    rows = load()
    tr = tax_rows(rows)
    plot_tax(tr)
    sp = speeds(SPEED.read_text(encoding="utf-8").splitlines()) if SPEED.exists() else {}
    b = board(rows, sp)
    write_space(b, datetime.datetime.now(datetime.timezone.utc).date().isoformat())
    print(f"kaydedildi: {PNG}, {SPACE}/index.html ({len(b)} satir)")


if __name__ == "__main__":
    main()
