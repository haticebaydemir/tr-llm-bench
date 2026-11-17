import asyncio
import json

import httpx

from trbench.load.generator import percentile, run, summarize


def sse(*chunks):
    lines = [f"data: {json.dumps(c, ensure_ascii=False)}" for c in chunks] + ["data: [DONE]"]
    return ("\n\n".join(lines) + "\n\n").encode()


def fake_server(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["stream"] and body["temperature"] == 0 and body["min_tokens"] == body["max_tokens"]
    data = sse(
        {"choices": [{"delta": {"content": "Ankara'ya "}}]},
        {"choices": [{"delta": {"content": "gittim."}}]},
        {"choices": [], "usage": {"completion_tokens": 7}},
    )
    return httpx.Response(200, content=data, headers={"content-type": "text/event-stream"})


def test_run_parses_stream_and_counts():
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake_server))
    results, wall = asyncio.run(run(["p"] * 5, "http://x", "m", concurrency=2, client=client))
    assert all(r.ok for r in results)
    assert results[0].out_tokens == 7
    assert results[0].out_words == 2                  # "Ankara'ya gittim."
    assert results[0].ttft_s is not None
    s = summarize(results, wall)
    assert s["requests"] == 5 and s["errors"] == 0


def test_server_error_is_recorded_not_raised():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    results, wall = asyncio.run(run(["p"] * 3, "http://x", "m", concurrency=3, client=client))
    s = summarize(results, wall)
    assert s["errors"] == 3 and "503" in results[0].error


def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10], 0.95) == 10


def test_concurrency_limit_is_respected():
    state = {"now": 0, "peak": 0}

    async def slow(request: httpx.Request) -> httpx.Response:
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        return fake_server(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    asyncio.run(run(["p"] * 12, "http://x", "m", concurrency=4, client=client))
    assert state["peak"] == 4
