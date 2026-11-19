"""Faz 4: eval setini bir modele sorar, puanlar, satir satir kaydeder.

OpenAI uyumlu her sunucuyla calisir (vLLM, llama.cpp server).
Gece gozetimsiz calisacagi icin: her cevap aninda yazilir, yeniden baslatinca kaldigi yerden.

Kullanim:
  python -m trbench.eval.run --url http://127.0.0.1:18010 --label qwen2.5-7b/q4_k_m
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx

from trbench.eval.build import OUT as EVAL
from trbench.eval.schema import Item
from trbench.eval.scorers import score
from trbench.provenance import file_sha256, provenance

OUT = Path("results/phase4.jsonl")
# Kategoriye gore cikti siniri: TR ayni icerik icin ~1.6x token harciyor (Faz 1),
# bu yuzden sinirlar Turkce'nin sigacagi kadar genis tutuldu; aksi halde TR cevaplar
# kesilir ve kalite kaybi diye olculurdu.
MAX_TOKENS = {"reading": 64, "knowledge": 64, "math": 768, "instruction": 768,
              "json": 160, "morphology": 32}


def load_items(path: Path = EVAL) -> list[Item]:
    return [Item(**json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines()]


def done_ids(path: Path, label: str) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["label"] == label and r.get("error") is None:
            out.add((r["id"], r["lang"]))
    return out


async def ask(client: httpx.AsyncClient, url: str, item: Item) -> tuple[str | None, str | None]:
    body = {"model": "x", "messages": [{"role": "user", "content": item.prompt}],
            "max_tokens": MAX_TOKENS[item.category], "temperature": 0, "seed": 0}
    try:
        r = await client.post(f"{url}/v1/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or "", None
    except (httpx.HTTPError, KeyError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"[:300]


async def run(items: list[Item], url: str, label: str, parallel: int, out: Path,
              client: httpx.AsyncClient | None = None, meta: dict | None = None) -> int:
    done = done_ids(out, label)
    todo = [i for i in items if (i.id, i.lang) not in done]
    own = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(1800.0))
    sem = asyncio.Semaphore(parallel)
    out.parent.mkdir(parents=True, exist_ok=True)

    async def one(item: Item) -> None:
        async with sem:
            t0 = time.perf_counter()
            resp, err = await ask(client, url, item)
            row = {"label": label, "id": item.id, "lang": item.lang, "category": item.category,
                   "correct": None if resp is None else score(item, resp),
                   "response": resp, "error": err,
                   "latency_s": round(time.perf_counter() - t0, 2), **(meta or {})}
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        await asyncio.gather(*(one(i) for i in todo))
    finally:
        if own:
            await client.aclose()
    return len(todo)


def accuracy(rows: list[dict]) -> dict[tuple[str, str, str], tuple[int, int]]:
    """(label, category, lang) -> (dogru, toplam). Hatali istekler sayilmaz."""
    acc: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("error") is not None:
            continue
        k = (r["label"], r["category"], r["lang"])
        acc[k][0] += int(bool(r["correct"]))
        acc[k][1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


def server_meta(url: str) -> dict:
    """llama.cpp /props: hangi surum ve hangi agirlik dosyasi cevapladi (vLLM'de yok, bos gecer)."""
    try:
        p = httpx.get(f"{url}/props", timeout=5).json()
    except (httpx.HTTPError, ValueError):
        return {}
    return {"server_build": p.get("build_info"), "weights_file": Path(p.get("model_path") or "").name or None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18010")
    ap.add_argument("--label", required=True, help="orn. qwen2.5-7b/q4_k_m")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="test icin: ilk N soru")
    a = ap.parse_args()
    items = load_items()
    if a.limit:
        items = items[: a.limit]
    t0 = time.time()
    meta = {**provenance(), "eval_sha": file_sha256(EVAL), **server_meta(a.url)}
    n = asyncio.run(run(items, a.url, a.label, a.parallel, OUT, meta=meta))
    rows = [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines()]
    print(f"{n} soru soruldu, {time.time() - t0:.0f} sn")
    for (lab, cat, lang), (ok, tot) in sorted(accuracy(rows).items()):
        if lab == a.label:
            print(f"  {cat:12s} {lang}  {ok:3d}/{tot:3d}  %{100 * ok / tot:5.1f}")


if __name__ == "__main__":
    main()
