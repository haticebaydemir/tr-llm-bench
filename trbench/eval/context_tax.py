"""Faz 7t: baglam penceresi vergisi (context window tax).

Ayni sohbet Turkcede tokenizer vergisi kadar daha cok token tutar. Bunun uzun baglamda uc bedeli var:
  1. pencere daha erken dolar        -> hesap: pencereye sigan icerik = pencere / vergi
  2. KV onbellegi daha cok yer tutar -> hesap: token x (katman x KV basi x bas boyutu x 2 x 2 bayt)
  3. uretim ve prefill yavaslar      -> olcum: llama-bench, GPU 2, Turkce ve Ingilizce esdeger derinlikte

Icerik uzunlugu Ingilizce token cinsinden E (1024 / 2048 / 4096); ayni icerik Turkcede E x vergi token.

  python -m trbench.eval.context_tax --plan     # scripts/gpu_context.sh icin: REPO AD DERINLIKLER
  python -m trbench.eval.context_tax            # results/gpu_context.jsonl -> README tablosu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trbench.config import load

R = Path("results")
CONFIG = "configs/gpu_llamacpp_cold.yaml"
CONTENT = (1024, 2048, 4096)          # icerik uzunlugu, Ingilizce token
WINDOW = 8192                          # karsilastirma icin baglam penceresi
BUDGET_GB = 7.5                        # GPU 2'deki pay (agirlik + KV + ~0,8 GB ek yuk)

# KV boyutu mimariden (HF config.json): katman, KV basi, bas boyutu. fp16 KV, K ve V.
ARCH = {
    "qwen2.5-7b": (28, 4, 128), "llama-3.1-8b": (32, 8, 128), "mistral-7b-v0.3": (32, 8, 128),
    "aya-expanse-8b": (32, 8, 128), "cosmos-tr-llama-8b": (32, 8, 128), "trendyol-7b": (32, 8, 128),
    "gemma-2-9b": (42, 8, 256),
}


def kv_bytes_per_token(model: str) -> int:
    layers, kv_heads, head_dim = ARCH[model]
    return layers * kv_heads * head_dim * 2 * 2


def taxes() -> dict[str, float]:
    from trbench.eval.gpu_report import tokenizer_stats
    return tokenizer_stats()["tax"].to_dict()


def depths(model: str, tax: float, weights_gb: float) -> list[int]:
    """Olculecek derinlikler: her icerik uzunlugu icin EN (E) ve TR (E x vergi); butceye sigmayan atlanir."""
    out = []
    for e in CONTENT:
        tr = round(e * tax)
        if weights_gb + 0.8 + tr * kv_bytes_per_token(model) / 1024**3 > BUDGET_GB:
            continue
        out += [e, tr]
    return sorted(set(out))


def plan(config: str = CONFIG) -> list[tuple[str, str, list[int]]]:
    cfg, tx = load(config), taxes()
    return [(m.repo, m.name, depths(m.name, tx[m.name], m.weights_gb_actual or 0.0)) for m in cfg.models]


def bench_rows(path: Path) -> list[dict]:
    """llama-bench jsonl: dosya adindan model, satirdan derinlik/uretim/prefill, tekrarlarin medyani."""
    import statistics

    from trbench.eval.gpu_report import gguf_model
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({"model": gguf_model(r["model_filename"]), "depth": r.get("n_depth", 0),
                     "n_prompt": r["n_prompt"], "n_gen": r["n_gen"],
                     "ts": statistics.median(r["samples_ts"])})
    return rows


def table(rows: list[dict], tx: dict[str, float], e: int = 4096) -> list[dict]:
    """Model basina, icerik uzunlugu e icin: EN ve TR derinlikte uretim hizi, prefill suresi, KV ve pencere."""
    out = []
    for m in sorted({r["model"] for r in rows}):
        tr_d = round(e * tx[m])
        dec = {r["depth"]: r["ts"] for r in rows if r["model"] == m and r["n_gen"] > 0}
        pre = {r["n_prompt"]: r["ts"] for r in rows if r["model"] == m and r["n_prompt"] > 0}
        if e not in dec or tr_d not in dec:
            continue
        kv = kv_bytes_per_token(m) / 1024**2
        out.append({
            "model": m, "tax": tx[m], "e": e, "tr_tokens": tr_d,
            "kv_en_mb": e * kv, "kv_tr_mb": tr_d * kv,
            "window_content": WINDOW / tx[m],
            "dec_en": dec[e], "dec_tr": dec[tr_d], "dec_drop": 1 - dec[tr_d] / dec[e],
            "pre_en_s": e / pre[e] if e in pre else None,
            "pre_tr_s": tr_d / pre[tr_d] if tr_d in pre else None,
        })
    return sorted(out, key=lambda t: t["tax"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="olculecek derinlikleri yazdir (gpu_context.sh)")
    ap.add_argument("--file", default=str(R / "gpu_context.jsonl"))
    ap.add_argument("--content", type=int, default=4096, help="icerik uzunlugu, Ingilizce token")
    a = ap.parse_args()
    if a.plan:
        for repo, name, d in plan():
            print(repo, name, ",".join(map(str, d)))
        return
    from trbench.eval.gpu_report import SHOW
    print(f"| Model | Tax | TR tokens for {a.content:,} EN tokens of content | KV cache EN / TR (MB) | "
          f"Content that fits {WINDOW:,} tokens (EN tokens) | Decode tok/s at EN / TR length | Change | "
          "Prefill EN / TR (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for t in table(bench_rows(Path(a.file)), taxes(), a.content):
        pre = (f"{t['pre_en_s']:.1f} / {t['pre_tr_s']:.1f}" if t["pre_en_s"] and t["pre_tr_s"] else "–")
        print(f"| {SHOW.get(t['model'], t['model'])} | {t['tax']:.2f} | {t['tr_tokens']:,} | "
              f"{t['kv_en_mb']:.0f} / {t['kv_tr_mb']:.0f} | {t['window_content']:,.0f} | "
              f"{t['dec_en']:.1f} / {t['dec_tr']:.1f} | {-t['dec_drop']:+.0%} | {pre} |")


if __name__ == "__main__":
    main()
