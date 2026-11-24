"""GPU sonuclari (V100, GPU 2'de ~8 GB pay): servis hizi, motor karsilastirmasi, girdi uzunlugu, maliyet.

Girdi:  results/gpu_llamacpp.jsonl, results/gpu_vllm.jsonl  (trbench.runner)
        results/gpu_length.jsonl                            (llama-bench -o jsonl)
        results/tokenizer_tax.csv                           (vergi ve TR token/kelime)
Cikti:  README icin markdown tablolar (stdout), docs/gpu_content.png

  python -m trbench.eval.gpu_report [--usd-per-hour 1.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

R = Path("results")
# runner'daki model adi -> tokenizer_tax.csv etiketi
LABEL = {"qwen2.5-7b": "Qwen2.5-7B", "llama-3.1-8b": "Llama-3.1-8B", "gemma-2-9b": "Gemma-2-9B",
         "mistral-7b-v0.3": "Mistral-7B-v0.3", "cosmos-tr-llama-8b": "Cosmos-TR-Llama-8B",
         "trendyol-7b": "Trendyol-7B", "aya-expanse-8b": "Aya-Expanse-8B"}
SHOW = {"qwen2.5-7b": "Qwen2.5-7B", "llama-3.1-8b": "Llama-3.1-8B", "gemma-2-9b": "Gemma-2-9B",
        "mistral-7b-v0.3": "Mistral-7B-v0.3", "cosmos-tr-llama-8b": "Cosmos-TR-Llama-8B",
        "trendyol-7b": "Trendyol-7B", "aya-expanse-8b": "Aya-Expanse-8B"}


def tokenizer_stats(csv: Path = R / "tokenizer_tax.csv") -> pd.DataFrame:
    """model -> tax (TR/EN token) ve fert (TR token/kelime)."""
    t = pd.read_csv(csv)
    tr = t[t["lang"] == "tur_Latn"].set_index("model")
    rows = {m: {"tax": tr.loc[lab, "tax_vs_eng"], "fert": tr.loc[lab, "fertility"]}
            for m, lab in LABEL.items() if lab in tr.index}
    return pd.DataFrame(rows).T


def serving(paths: list[Path]) -> pd.DataFrame:
    """Tekrarlarin medyani: engine, model, lang, concurrency -> tok/s, ttft, tepe bellek."""
    import json

    from trbench.results import pick
    rows = [json.loads(x) for p in paths if p.exists()
            for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = pick([r for r in rows if r.get("status") == "ok"],
                ("engine", "model", "quant"), ("lang", "concurrency", "repeat"))
    df = pd.DataFrame(rows)
    if "words_per_s" not in df:
        df["words_per_s"] = float("nan")
    g = df.groupby(["engine", "model", "lang", "concurrency"])
    out = g.agg(tps=("tokens_per_s", "median"), wps=("words_per_s", "median"), ttft=("ttft_p50_ms", "median"),
                ttft_max=("ttft_p50_ms", "max"), errors=("errors", "sum"),
                peak=("gpu2_peak_mib", "max"), idle=("gpu2_idle_mib", "min"), n=("repeat", "count"))
    return out.reset_index()


def top_conc(s: pd.DataFrame) -> pd.DataFrame:
    """Her (engine, model) icin olculen en yuksek esanlilik."""
    mx = s.groupby(["engine", "model"])["concurrency"].transform("max")
    return s[s["concurrency"] == mx]


def model_table(s: pd.DataFrame, tok: pd.DataFrame, usd_h: float) -> pd.DataFrame:
    lc = s[s["engine"] == "llamacpp"]
    c1 = lc[(lc["concurrency"] == 1) & (lc["lang"] == "tr")].set_index("model")
    top = top_conc(lc)
    tr = top[top["lang"] == "tr"].set_index("model")
    en = top[top["lang"] == "en"].set_index("model")
    t = pd.DataFrame({
        "c1_tr": c1["tps"], "conc": tr["concurrency"], "top_tr": tr["tps"], "top_en": en["tps"],
        "wps_tr": tr["wps"],
        "ttft_tr": tr["ttft"], "peak_gb": lc.groupby("model")["peak"].max() / 1024,
        "ours_gb": (lc.groupby("model")["peak"].max() - lc.groupby("model")["idle"].min()) / 1024,
    })
    t = t.join(tok)
    t["content"] = t["top_tr"] / t["tax"]                        # Ingilizce esdegeri icerik tok/s
    t["words_s"] = t["top_tr"] / t["fert"]                       # TR kelime/s (FLORES token/kelime ile)
    t["usd_per_1k_words"] = usd_h / 3600 / t["words_s"] * 1000
    return t.sort_values("content", ascending=False)


def engine_table(s: pd.DataFrame, model: str = "mistral-7b-v0.3") -> pd.DataFrame:
    m = s[s["model"] == model]
    p = m.pivot_table(index=["concurrency", "lang"], columns="engine", values=["tps", "ttft"])
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p["vllm_vs_llamacpp"] = p["tps_vllm"] / p["tps_llamacpp"] - 1
    return p.reset_index()


def gguf_model(path: str) -> str:
    """'/.../gguf/llama-3.1-8b/llama-3.1-8b-q4_k_m.gguf' -> 'llama-3.1-8b'; Qwen'in resmi klasoru ayri."""
    d = path.replace("\\", "/").split("/")[-2].lower()
    return "qwen2.5-7b" if d.startswith("qwen__qwen2.5-7b") else d


def length_table(path: Path = R / "gpu_length.jsonl") -> pd.DataFrame:
    """Tekrarlarin MEDYANI: GPU servisle paylasildigi icin tek tekrarda ani dusus olabiliyor
    (or. 37, 37, 21 tok/s); ortalama bunu sonuca tasir, medyan tasimaz."""
    x = pd.read_json(path, lines=True)
    x["model"] = x["model_filename"].map(gguf_model)
    x["ts"] = x["samples_ts"].map(lambda s: float(pd.Series(s).median()))
    tg = x[x["n_gen"] > 0].pivot_table(index="model", columns="n_depth", values="ts")
    pp = x[x["n_prompt"] == 4096].pivot_table(index="model", columns="n_depth", values="ts")
    return pd.DataFrame({"tg_0": tg[0], "tg_4096": tg[4096], "tg_drop": 1 - tg[4096] / tg[0],
                         "pp4096_0": pp[0], "pp4096_at4096": pp[4096]})


def plot(t: pd.DataFrame, path: Path = Path("docs/gpu_content.png")) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from trbench.tokenizer_tax.plot import GRID, INK, INK2, SERIES, SURFACE

    d = t.sort_values("content")
    names = [SHOW[m] for m in d.index]
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(d))
    ax.barh([i + 0.2 for i in y], d["top_tr"], height=0.36, color="#c3c2b7", label="Raw Turkish tokens/s")
    ax.barh([i - 0.2 for i in y], d["content"], height=0.36, color=SERIES,
            label="Turkish content/s (tokens ÷ tokenizer tax)")
    for i, v in zip(y, d["content"], strict=True):
        ax.text(v + 2, i - 0.2, f"{v:.0f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(list(y), names, color=INK, fontsize=9)
    ax.set_xlabel(f"Tokens per second, {int(d['conc'].max())} concurrent requests "
                  "(Gemma: 2, memory budget)", color=INK2)
    ax.tick_params(colors=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    same = d[d["conc"] == d["conc"].max()]                       # Gemma (c=2) ayni kosulda degil -> oran disi
    hi, lo = same["content"].max(), same["content"].min()
    fig.suptitle(f"At {int(d['conc'].max())} concurrent requests raw speed is similar, "
                 f"but Turkish content per second differs {hi / lo:.1f}×",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_title("llama.cpp Q4_K_M on one V100 with an ~8 GB memory budget; prompt cache off; median of 3 repeats",
                 loc="left", color=INK2, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd-per-hour", type=float, default=1.0,
                    help="GPU saat fiyati varsayimi; maliyet bununla dogrusal olceklenir")
    ap.add_argument("--llamacpp-file", default="gpu_llamacpp_cold.jsonl",
                    help="results/ altindaki llama.cpp dosyasi (orn. gpu_llamacpp_cold.jsonl: onbellek kapali)")
    a = ap.parse_args()
    s = serving([R / a.llamacpp_file, R / "gpu_vllm.jsonl"])
    print(f"kosu: {int(s['n'].sum())}, hata: {int(s['errors'].sum())}, "
          f"GPU 2 toplam tepe: {s['peak'].max() / 1024:.1f} GB\n")
    t = model_table(s, tokenizer_stats(), a.usd_per_hour)
    print("| Model | c=1 TR tok/s | Top c | TR tok/s | EN tok/s | TR words/s (measured) | TR content tok/s "
          "| TTFT p50 TR (ms) "
          f"| GPU 2 peak (GB) | $ per 1k TR words @ ${a.usd_per_hour:g}/h |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m, r in t.iterrows():
        print(f"| {SHOW[m]} | {r.c1_tr:.0f} | {int(r.conc)} | {r.top_tr:.0f} | {r.top_en:.0f} | "
              f"{r.wps_tr:.0f} | **{r.content:.0f}** | {r.ttft_tr:.0f} | {r.peak_gb:.1f} | {r.usd_per_1k_words:.4f} |")
    e = engine_table(s)
    print("\n| Concurrency | Lang | llama.cpp tok/s | vLLM tok/s | vLLM vs llama.cpp | "
          "TTFT p50 llama.cpp / vLLM (ms) |")
    print("|---|---|---|---|---|---|")
    for _, r in e.iterrows():
        print(f"| {r.concurrency} | {r.lang.upper()} | {r.tps_llamacpp:.0f} | {r.tps_vllm:.0f} | "
              f"{r.vllm_vs_llamacpp:+.0%} | {r.ttft_llamacpp:.0f} / {r.ttft_vllm:.0f} |")
    if (R / "gpu_length.jsonl").exists():
        lt = length_table()
        print("\n| Model | Decode tok/s, empty context | at 4,096 tokens of context | Change | "
              "Prefill tok/s (4,096-token prompt) |")
        print("|---|---|---|---|---|")
        for m, r in lt.sort_values("tg_drop").iterrows():
            print(f"| {SHOW.get(m, m)} | {r.tg_0:.1f} | {r.tg_4096:.1f} | {-r.tg_drop:+.0%} | {r.pp4096_0:.0f} |")
    print(f"\nkaydedildi: {plot(t)}")


if __name__ == "__main__":
    main()
