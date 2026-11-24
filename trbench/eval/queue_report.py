"""Faz 6d: Turkce TTFT artisinin nedeni (kuyruk mu, token sayisi mi?).

  python -m trbench.eval.queue_report                 # results/gpu_6d.jsonl
  python -m trbench.eval.queue_report --samples       # zorlanmis ciktilarin ornekleri

  python -m trbench.eval.queue_report --file results/gpu_6d_mnbt.jsonl   # Faz 7a: adim basina prompt butcesi
  python -m trbench.eval.queue_report --file results/gpu_6d_mnbt.jsonl --by-repeat   # 7k: tekrar 0 farkli mi

Her satir: KV butcesi (vLLM gpu_memory_utilization), adim basina prompt butcesi (--max-num-batched-tokens),
prompt uzunlugu (dogal ya da TR/EN esit), dil, esanlilik -> tekrarlarin medyani;
kuyruk vLLM /metrics'ten (kosu boyunca bekleyen istek).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from trbench.results import pick

R = Path("results")


def batched_tokens(r: dict) -> int | None:
    """--max-num-batched-tokens degeri; eski satirlarda extra_args yok -> None."""
    a = r.get("extra_args") or []
    for i, x in enumerate(a[:-1]):
        if x == "--max-num-batched-tokens":
            return int(a[i + 1])
    return None


def variant(r: dict) -> str:
    """Prefix cache ve /metrics sorgusu: 7a/7k kontrolleri. Eski 6d satirlari: cache acik, sorgu acik."""
    off = "--no-enable-prefix-caching" in (r.get("extra_args") or []) or r.get("cache_prompt") is False
    pc = "off" if off else "on"
    qm = "on" if r.get("queue_metrics", True) else "off"
    return f"cache {pc}, metrics {qm}"


def load(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [{**r, "batched_tokens": batched_tokens(r), "variant": variant(r)}
            for r in rows if r.get("status") == "ok"]
    return pick(rows, ("engine", "model", "quant", "gpu_memory_utilization", "batched_tokens", "variant",
                       "prompt_tokens_target"), ("lang", "concurrency", "repeat"))


def table(rows: list[dict]) -> list[dict]:
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        g[(r.get("gpu_memory_utilization"), r.get("batched_tokens"), r.get("variant", ""),
           r.get("prompt_tokens_target"), r["lang"], r["concurrency"])].append(r)

    def med(rs: list[dict], k: str) -> float | None:
        v = [r[k] for r in rs if r.get(k) is not None]
        return st.median(v) if v else None

    out = []
    for (u, bt, var, pt, lang, c), rs in sorted(g.items(), key=lambda kv: (
            -(kv[0][0] or 0), kv[0][1] or 0, kv[0][2], kv[0][3] or 0, kv[0][5], kv[0][4])):
        out.append({"util": u, "batched": bt, "variant": var, "prompt": pt or "natural", "lang": lang, "c": c,
                    "n": len(rs), "ttft_by_repeat": [r.get("ttft_p50_ms") for r in sorted(rs, key=lambda r: r["repeat"])],
                    "tps": med(rs, "tokens_per_s"), "ttft50": med(rs, "ttft_p50_ms"),
                    "ttft95": med(rs, "ttft_p95_ms"), "ttft99": med(rs, "ttft_p99_ms"),
                    "ptok": med(rs, "prompt_tokens_mean"),
                    "wait_max": max((r.get("queue_waiting_max") or 0) for r in rs),
                    "wait_mean": med(rs, "queue_waiting_mean"),
                    "preempt": sum((r.get("preemptions") or 0) for r in rs)})
    return out


def _f(x: float | None, fmt: str = "{:.0f}") -> str:
    return "–" if x is None else fmt.format(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=R / "gpu_6d.jsonl")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--by-repeat", action="store_true", help="TTFT p50'yi tekrar tekrar goster (7k)")
    a = ap.parse_args()
    if a.samples:
        for r in load(R / "gpu_6d_samples.jsonl"):
            print(f"\n=== {r['model']} {r['lang']} (tok/s {r['tokens_per_s']:.0f}, kelime/s {r['words_per_s']:.0f})")
            for s in r.get("sample_outputs", []):
                print("---", repr(s))
        return
    rows = table(load(a.file))
    show_bt = any(t["batched"] is not None for t in rows)        # eski dosyada sutun yok: tablo aynen kalir
    if a.by_repeat:
        print("| Batched tokens | Variant | Prompt tokens | Lang | c | TTFT p50 by repeat (ms) |")
        print("|---|---|---|---|---|---|")
        for t in rows:
            reps = " / ".join(_f(x) for x in t["ttft_by_repeat"])
            print(f"| {_f(t['batched'])} | {t['variant']} | {t['prompt']} | {t['lang'].upper()} | {t['c']} | {reps} |")
        return
    print("| KV budget (util) | " + ("Batched tokens | Variant | " if show_bt else "") + "Prompt tokens | Lang | c | n | "
          "tok/s | TTFT p50 / p95 / p99 (ms; 16-32 requests per run, so p99 ~ max) | Prompt tok (mean) | Waiting max / mean | Preemptions |")
    print("|---|" + ("---|---|" if show_bt else "") + "---|---|---|---|---|---|---|---|---|")
    for t in rows:
        bt = f"{_f(t['batched'])} | {t['variant']} | " if show_bt else ""
        print(f"| {t['util']} | {bt}{t['prompt']} | {t['lang'].upper()} | {t['c']} | {t['n']} | {_f(t['tps'])} | "
              f"{_f(t['ttft50'])} / {_f(t['ttft95'])} / {_f(t['ttft99'])} | {_f(t['ptok'])} | "
              f"{t['wait_max']:.0f} / {_f(t['wait_mean'], '{:.1f}')} | {t['preempt']:.0f} |")


if __name__ == "__main__":
    main()
