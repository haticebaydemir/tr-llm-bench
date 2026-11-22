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

GPU_Q = Path("results/gpu_quality.jsonl")
GPU_S = Path("results/gpu_llamacpp.jsonl")
TOK = Path("results/tokenizer_tax.csv")
# GPU olcumundeki ad -> tokenizer_tax.csv etiketi (ayni tokenizer'i paylasanlar ayni etiketi alir)
TOK_LABEL = {"qwen3-8b": "Qwen3-8B", "qwen3-4b": "Qwen3-8B", "qwen2.5-7b": "Qwen2.5-7B",
             "gemma-3-4b": "Gemma-3-12B", "gemma-2-9b": "Gemma-2-9B", "llama-3.1-8b": "Llama-3.1-8B",
             "llama-3.2-3b": "Llama-3.1-8B", "cosmos-tr-llama-8b": "Cosmos-TR-Llama-8B",
             "aya-expanse-8b": "Aya-Expanse-8B", "trendyol-7b": "Trendyol-7B", "mistral-7b-v0.3": "Mistral-7B-v0.3",
             "ministral-8b": "Ministral-8B", "phi-4-mini": "Phi-4-mini", "granite-3.3-8b": "Granite-3.3-8B"}
CATS = ("reading", "math", "knowledge", "instruction", "json")

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


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def tok_tax(csv: Path = TOK) -> dict[str, float]:
    import csv as _csv
    if not csv.exists():
        return {}
    with csv.open(encoding="utf-8") as f:
        t = {r["model"]: float(r["tax_vs_eng"]) for r in _csv.DictReader(f) if r["lang"] == "tur_Latn"}
    return {m: t[lab] for m, lab in TOK_LABEL.items() if lab in t}


def gpu_speed(rows: list[dict], field: str = "tokens_per_s") -> dict[str, float]:
    """Turkce, olculen en yuksek esanlilikta, tekrarlarin medyani (tok/s ya da olculen kelime/s)."""
    import statistics
    by: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        if r.get("status") == "ok" and r["lang"] == "tr" and r.get(field) is not None:
            by.setdefault(r["model"], {}).setdefault(r["concurrency"], []).append(r[field])
    return {m: statistics.median(d[max(d)]) for m, d in by.items()}


def gpu_board(q: list[dict], s: list[dict], tax_: dict[str, float]) -> list[dict]:
    """14 model, GPU Q4_K_M: TR/EN dogruluk (+-%95 normal yaklasim), kategoriler, hiz."""
    acc: dict[str, dict[str, list[int]]] = {}
    for r in q:
        if r.get("correct") is None or r["id"] in TRANSLATION_ERRORS:
            continue
        m = r["label"].split("/")[0]
        keys = ["morph"] if r["category"] == "morphology" else [r["lang"], f"{r['lang']}_{r['category']}"]
        for k in keys:
            acc.setdefault(m, {}).setdefault(k, []).append(int(r["correct"]))
    sp = gpu_speed(s)
    wp = gpu_speed(s, "words_per_s")
    out = []
    for m, d in acc.items():
        def pct(k, d=d):
            v = d.get(k)
            return round(100 * sum(v) / len(v), 1) if v else None
        tr = d.get("tr", [])
        p = sum(tr) / len(tr) if tr else 0
        ci = round(196 * (p * (1 - p) / len(tr)) ** 0.5, 1) if tr else None
        row = {"model": m, "tr": pct("tr"), "tr_ci": ci, "en": pct("en"), "morph": pct("morph"),
               **{c: pct(f"tr_{c}") for c in CATS}, "tok_tax": tax_.get(m),
               "gpu_tps": round(sp[m]) if m in sp else None,
               "gpu_wps": round(wp[m]) if m in wp else None}
        row["tr_content_tps"] = round(sp[m] / tax_[m]) if m in sp and m in tax_ else None
        out.append(row)
    return sorted(out, key=lambda r: -(r["tr"] or 0))


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TR-LLM Bench leaderboard</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1240px;margin:2rem auto;padding:0 1rem;color:#1d1d1f}}
h1{{font-size:1.5rem;margin:0 0 .25rem}} h2{{font-size:1.15rem;margin:2rem 0 .25rem}} p{{color:#555;margin:.25rem 0 1rem}}
.wrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}}
th,td{{padding:.4rem .5rem;border-bottom:1px solid #e5e5e5;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} td.q,th.q{{text-align:left}}
th{{cursor:pointer;user-select:none;background:#fafafa;font-weight:600}} th:hover{{background:#f0f0f0}}
tr:hover td{{background:#f7f9fc}} a{{color:#2a6fdb}} small,.ci{{color:#777}}
</style></head><body>
<h1>TR-LLM Bench leaderboard</h1>
<p>Open LLMs on Turkish vs English: accuracy on 242 paired items (8 mistranslated items removed),
Turkish morphology, tokenizer tax and serving speed. Greedy decoding, deterministic scoring, no LLM judge.
Click a column to sort. <a href="https://github.com/haticebaydemir/tr-llm-bench" target="_blank">Code &amp; method</a> ·
<a href="https://huggingface.co/datasets/HaticeB/tr-llm-bench-eval" target="_blank">Eval set</a></p>

<h2>Models (Q4_K_M, GPU)</h2>
<p>14 models, llama.cpp on one V100. ± is the 95% interval of Turkish accuracy (about ±5 points):
neighbouring rows are usually not distinguishable. Category columns are Turkish accuracy.
Speeds are Turkish, at 8 concurrent requests (Gemma-2: 2; speed measured for 7 models):
<b>GPU tok/s</b> = raw tokens per second; <b>TR content tok/s</b> = GPU tok/s ÷ tokenizer tax,
i.e. English-equivalent tokens of content per second.</p>
<div class="wrap"><table class="s"><thead><tr><th>Model</th><th>TR %</th><th>EN %</th><th>TR−EN</th>
<th>Morph.</th><th>Reading</th><th>Math</th><th>Knowl.</th><th>Instr.</th><th>JSON</th>
<th>Tok. tax</th><th>GPU tok/s</th><th>TR content tok/s</th></tr></thead><tbody>
{gpu_rows}
</tbody></table></div>

<h2>Quantization (CPU)</h2>
<p>Seven models at fp16, Q8_0 and Q4_K_M, llama.cpp <code>b12470</code> on 16 CPU cores.
<b>TR content tok/s</b> = CPU tok/s ÷ tokenizer tax.</p>
<div class="wrap"><table class="s"><thead><tr><th>Model</th><th class="q">Quant</th><th>TR %</th><th>EN %</th><th>TR−EN</th>
<th>Morph.</th><th>Tok. tax</th><th>CPU tok/s</th><th>TR content tok/s</th></tr></thead><tbody>
{rows}
</tbody></table></div>
<p><small>Updated {date}.</small></p>
<script>
document.querySelectorAll('table.s').forEach(t=>t.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{{
 const rs=[...t.tBodies[0].rows],asc=th.dataset.asc!=='1';
 rs.sort((a,b)=>{{let x=a.cells[i].dataset.v,y=b.cells[i].dataset.v;
  if(x!==undefined||y!==undefined){{x=parseFloat(x);y=parseFloat(y);x=isNaN(x)?-1e9:x;y=isNaN(y)?-1e9:y;return asc?x-y:y-x}}
  x=a.cells[i].textContent;y=b.cells[i].textContent;return asc?x.localeCompare(y):y.localeCompare(x)}});
 th.dataset.asc=asc?'1':'0';rs.forEach(r=>t.tBodies[0].appendChild(r))}}));
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
quality, tokenizer tax and serving speed of 14 open LLMs, plus a quantization comparison on CPU.
"""


def cell(v, fmt="{:.1f}") -> str:
    return "<td>–</td>" if v is None else f'<td data-v="{v}">{fmt.format(v)}</td>'


def write_space(b: list[dict], date: str, g: list[dict] | None = None) -> None:
    trs = []
    for r in sorted(b, key=lambda r: -(r["tr"] or 0)):
        gap = None if r["tr"] is None or r["en"] is None else round(r["tr"] - r["en"], 1)
        trs.append("<tr><td>" + html.escape(r["model"]) + '</td><td class="q">' + r["quant"].upper() + "</td>"
                   + cell(r["tr"]) + cell(r["en"]) + cell(gap, "{:+.1f}") + cell(r["morph"])
                   + cell(r["tok_tax"], "{:.2f}×") + cell(r["cpu_tps"]) + cell(r["tr_content_tps"]) + "</tr>")
    grs = []
    for r in g or []:
        gap = None if r["tr"] is None or r["en"] is None else round(r["tr"] - r["en"], 1)
        ci = f' <span class="ci">±{r["tr_ci"]:.0f}</span>' if r["tr_ci"] else ""
        trc = "<td>–</td>" if r["tr"] is None else f'<td data-v="{r["tr"]}">{r["tr"]:.1f}{ci}</td>'
        grs.append("<tr><td>" + html.escape(r["model"]) + "</td>" + trc
                   + cell(r["en"]) + cell(gap, "{:+.1f}") + cell(r["morph"])
                   + "".join(cell(r[c], "{:.0f}") for c in CATS)
                   + cell(r["tok_tax"], "{:.2f}×") + cell(r["gpu_tps"], "{:.0f}") + cell(r["tr_content_tps"], "{:.0f}")
                   + "</tr>")
    SPACE.mkdir(exist_ok=True)
    (SPACE / "index.html").write_text(PAGE.format(rows="\n".join(trs), gpu_rows="\n".join(grs), date=date),
                                      encoding="utf-8")
    (SPACE / "README.md").write_text(CARD, encoding="utf-8")
    (SPACE / "data.json").write_text(json.dumps({"gpu_q4": g or [], "cpu_quant": b}, indent=1), encoding="utf-8")


def main() -> None:
    import datetime
    rows = load()
    tr = tax_rows(rows)
    plot_tax(tr)
    sp = speeds(SPEED.read_text(encoding="utf-8").splitlines()) if SPEED.exists() else {}
    b = board(rows, sp)
    from trbench.eval.gpu_quality_report import load as load_q
    from trbench.results import pick
    s_rows = pick([r for r in _jsonl(GPU_S) if r.get("status") == "ok"],
                  ("engine", "model", "quant"), ("lang", "concurrency", "repeat"))
    g = gpu_board(load_q(GPU_Q), s_rows, tok_tax())
    write_space(b, datetime.datetime.now(datetime.timezone.utc).date().isoformat(), g)
    print(f"kaydedildi: {PNG}, {SPACE}/index.html (GPU {len(g)} model, CPU {len(b)} satir)")


if __name__ == "__main__":
    main()
