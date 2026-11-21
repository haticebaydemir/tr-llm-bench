"""GPU dogruluk olcumu (results/gpu_quality.jsonl): model siralamasi ve GPU-CPU tutarliligi.

  python -m trbench.eval.gpu_quality_report

Paralel (TR+EN) sorular, ceviri hatalari (analysis.TRANSLATION_ERRORS) cikarilarak; morfoloji sadece TR.
GPU-CPU: ayni GGUF dosyasi CPU'da (results/phase4.jsonl) da olculduyse, sorularin yuzde kacinda
dogru/yanlis sonucu ayni.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from trbench.eval.analysis import TRANSLATION_ERRORS

GPU = Path("results/gpu_quality.jsonl")
CPU = Path("results/phase4.jsonl")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r.get("correct") is not None]


def table(rows: list[dict]) -> dict[str, dict]:
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if r["id"] in TRANSLATION_ERRORS:
            continue
        key = "morph" if r["category"] == "morphology" else r["lang"]
        acc[r["label"]][key][0] += int(r["correct"])
        acc[r["label"]][key][1] += 1
    return {lab: {k: v[0] / v[1] for k, v in d.items()} for lab, d in acc.items()}


def agreement(gpu: list[dict], cpu: list[dict]) -> dict[str, tuple[float, int]]:
    c = {(r["label"], r["id"], r["lang"]): bool(r["correct"]) for r in cpu}
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in gpu:
        k = (r["label"], r["id"], r["lang"])
        if k in c:
            out[r["label"]][0] += int(c[k] == bool(r["correct"]))
            out[r["label"]][1] += 1
    return {lab: (a / n, n) for lab, (a, n) in out.items()}


def main() -> None:
    gpu, cpu = load(GPU), load(CPU)
    t, cp, ag = table(gpu), table(cpu), agreement(gpu, cpu)
    print(f"GPU satiri: {len(gpu)}, model: {len(t)}\n")
    print("| Model | TR | EN | TR - EN | Morphology (TR) | CPU TR | Same result as CPU |")
    print("|---|---|---|---|---|---|---|")
    for lab, d in sorted(t.items(), key=lambda kv: -kv[1].get("tr", 0)):
        name = lab.split("/")[0]
        c = cp.get(lab, {})
        a = ag.get(lab)
        print(f"| {name} | {d.get('tr', 0):.0%} | {d.get('en', 0):.0%} | {d.get('tr', 0) - d.get('en', 0):+.0%} | "
              f"{d.get('morph', 0):.0%} | {c['tr']:.0%} | {a[0]:.0%} of {a[1]} |" if a else
              f"| {name} | {d.get('tr', 0):.0%} | {d.get('en', 0):.0%} | {d.get('tr', 0) - d.get('en', 0):+.0%} | "
              f"{d.get('morph', 0):.0%} | – | – |")


if __name__ == "__main__":
    main()
