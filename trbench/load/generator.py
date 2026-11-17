"""Esanli yuk ureteci: OpenAI uyumlu /v1/chat/completions endpoint'ine stream istekleri.

Her istek icin olcer: TTFT (ilk token), toplam sure, uretilen token ve kelime.
Kullanim (sunucuda):
  python -m trbench.load.generator --url http://127.0.0.1:18000 --model smoke \\
      --concurrency 4 --n 32 --lang tr
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass

import httpx

from trbench.metrics.text import count_words


@dataclass
class Result:
    ok: bool
    ttft_s: float | None
    latency_s: float
    out_tokens: int
    out_words: int
    error: str = ""


def payload(model: str, prompt: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # Cikti uzunlugunu sabitle: TR ve EN ayni token sayisi uretsin ki
        # hiz farki 'kim daha kisa cevap verdi'den degil, dilden gelsin.
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def one_request(client: httpx.AsyncClient, url: str, body: dict) -> Result:
    start = time.perf_counter()
    ttft, text, tokens = None, [], 0
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    tokens = chunk["usage"].get("completion_tokens", tokens)
                for ch in chunk.get("choices", []):
                    piece = ch.get("delta", {}).get("content") or ""
                    if piece and ttft is None:
                        ttft = time.perf_counter() - start
                    text.append(piece)
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        return Result(False, None, time.perf_counter() - start, 0, 0, f"{type(e).__name__}: {e}")
    out = "".join(text)
    return Result(True, ttft, time.perf_counter() - start, tokens, count_words(out))


async def run(prompts: list[str], url: str, model: str, concurrency: int,
              max_tokens: int = 256, client: httpx.AsyncClient | None = None
              ) -> tuple[list[Result], float]:
    """Kapali dongu: ayni anda en fazla `concurrency` istek; biri bitince digeri baslar."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(600.0))
    sem = asyncio.Semaphore(concurrency)

    async def guarded(p: str) -> Result:
        async with sem:
            return await one_request(client, url, payload(model, p, max_tokens))

    t0 = time.perf_counter()
    try:
        results = await asyncio.gather(*(guarded(p) for p in prompts))
    finally:
        if own:
            await client.aclose()
    return list(results), time.perf_counter() - t0


def percentile(xs: list[float], q: float) -> float:
    """Dogrusal ara degerli yuzdelik (numpy 'linear' ile ayni)."""
    if not xs:
        return math.nan
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list[Result], wall_s: float) -> dict:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    lat = [r.latency_s for r in ok]
    toks = sum(r.out_tokens for r in ok)
    words = sum(r.out_words for r in ok)
    return {
        "requests": len(results), "errors": len(results) - len(ok),
        "ttft_p50_ms": percentile(ttft, 0.5) * 1000, "ttft_p95_ms": percentile(ttft, 0.95) * 1000,
        "latency_p50_s": percentile(lat, 0.5), "latency_p95_s": percentile(lat, 0.95),
        "tokens_per_s": toks / wall_s if wall_s else math.nan,
        "words_per_s": words / wall_s if wall_s else math.nan,
        "wall_s": wall_s,
    }


def main() -> None:
    from trbench.eval.build import OUT as EVAL

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--lang", choices=["tr", "en"], default="tr")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()

    items = [json.loads(x) for x in EVAL.read_text(encoding="utf-8").splitlines()]
    prompts = [i["prompt"] for i in items if i["lang"] == a.lang][: a.n]
    results, wall = asyncio.run(run(prompts, a.url, a.model, a.concurrency, a.max_tokens))
    summary = summarize(results, wall)
    print(json.dumps({"config": vars(a), **summary}, indent=2, ensure_ascii=False))
    errs = [asdict(r) for r in results if not r.ok][:3]
    if errs:
        print("ornek hatalar:", errs)


if __name__ == "__main__":
    main()
