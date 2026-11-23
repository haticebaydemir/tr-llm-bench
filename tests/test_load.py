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


def test_parse_metrics_sums_labels_and_finds_preemptions():
    from trbench.load.generator import parse_metrics
    txt = """# HELP x
vllm:num_requests_waiting{engine="0",model_name="m"} 3.0
vllm:num_requests_running{engine="0",model_name="m"} 5.0
vllm:num_preemptions_total{engine="0",model_name="m"} 2.0
vllm:num_preemptions_created{engine="0"} 1700000000.0
"""
    m = parse_metrics(txt)
    assert m == {"waiting": 3.0, "running": 5.0, "preemptions": 2.0}


def test_queue_watch_records_waiting_and_preemption_delta():
    calls = {"n": 0}

    def server(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            calls["n"] += 1
            w = 4 if calls["n"] > 1 else 0
            return httpx.Response(200, text=f"vllm:num_requests_waiting {w}\nvllm:num_preemptions_total {calls['n']}\n")
        return fake_server(request)

    async def go():
        client = httpx.AsyncClient(transport=httpx.MockTransport(server))
        q = {}

        async def slow_run():
            await asyncio.sleep(0.6)
            return await run(["p"] * 2, "http://x", "m", 1, client=client, queue=q)
        await slow_run()
        return q

    q = asyncio.run(go())
    assert q["queue_samples"] >= 1 and q["queue_waiting_max"] >= 0


def test_summary_has_p99_prompt_tokens_and_samples():
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake_server))
    results, wall = asyncio.run(run(["p"] * 3, "http://x", "m", concurrency=1, client=client))
    s = summarize(results, wall)
    assert "ttft_p99_ms" in s and s["sample_outputs"][0] == "Ankara'ya gittim."


def test_payload_extra_fields_for_llamacpp_cache():
    from trbench.load.generator import payload
    assert "cache_prompt" not in payload("m", "p", 16)
    assert payload("m", "p", 16, {"cache_prompt": False})["cache_prompt"] is False
