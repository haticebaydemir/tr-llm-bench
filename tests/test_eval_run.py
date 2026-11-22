import asyncio
import json

import httpx

from trbench.eval.run import accuracy, done_ids, run
from trbench.eval.schema import Item


def items():
    return [
        Item(id="k/1", category="knowledge", lang="tr", prompt="p", answer="B", scorer="choice",
             source=""),
        Item(id="k/1", category="knowledge", lang="en", prompt="p", answer="B", scorer="choice",
             source=""),
        Item(id="m/1", category="math", lang="tr", prompt="p", answer="64", scorer="number",
             source=""),
    ]


def server(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["temperature"] == 0 and body["max_tokens"] in (64, 768)
    text = "Cevap: B" if body["max_tokens"] == 64 else "...\nCevap: 46"
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_run_scores_and_resumes(tmp_path):
    out = tmp_path / "p4.jsonl"
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    n = asyncio.run(run(items(), "http://x", "m/q4", 2, out, client=client))
    assert n == 3
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    acc = accuracy(rows)
    assert acc[("m/q4", "knowledge", "tr")] == (1, 1)
    assert acc[("m/q4", "math", "tr")] == (0, 1)              # 46 != 64
    # ikinci kez: hepsi bitmis, hicbir sey sorulmaz
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    assert asyncio.run(run(items(), "http://x", "m/q4", 2, out, client=client)) == 0
    assert len(done_ids(out, "m/q4")) == 3


def test_errors_are_retried_and_not_scored(tmp_path):
    out = tmp_path / "p4.jsonl"
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    asyncio.run(run(items()[:1], "http://x", "m/q4", 1, out, client=client))
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["error"] and rows[0]["correct"] is None
    assert accuracy(rows) == {}
    assert done_ids(out, "m/q4") == set()                     # hata -> yeniden denenecek


def test_resume_depends_on_experiment_id(tmp_path):
    """Eval seti / agirlik / kod degisince (yeni exp) eski cevaplar yeniden kullanilmaz."""
    out = tmp_path / "p4.jsonl"
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    assert asyncio.run(run(items(), "http://x", "m/q4", 2, out, client=client, meta={"exp": "a"})) == 3
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    assert asyncio.run(run(items(), "http://x", "m/q4", 2, out, client=client, meta={"exp": "a"})) == 0
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    assert asyncio.run(run(items(), "http://x", "m/q4", 2, out, client=client, meta={"exp": "b"})) == 3
    assert len(done_ids(out, "m/q4", "a")) == 3 and len(done_ids(out, "m/q4", "b")) == 3
