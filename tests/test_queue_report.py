from trbench.eval.queue_report import table


def test_table_groups_by_budget_prompt_lang_conc():
    rows = [{"gpu_memory_utilization": 0.2, "prompt_tokens_target": None, "lang": "tr", "concurrency": 8,
             "repeat": i, "tokens_per_s": 100 + i, "ttft_p50_ms": 2000, "ttft_p95_ms": 2500, "ttft_p99_ms": 2600,
             "prompt_tokens_mean": 300, "queue_waiting_max": 3, "queue_waiting_mean": 1.0, "preemptions": 1}
            for i in range(3)]
    rows += [{**rows[0], "lang": "en", "queue_waiting_max": 0, "preemptions": 0}]
    t = table(rows)
    tr = next(x for x in t if x["lang"] == "tr")
    assert tr["n"] == 3 and tr["tps"] == 101 and tr["wait_max"] == 3 and tr["preempt"] == 3
    assert next(x for x in t if x["lang"] == "en")["wait_max"] == 0


def test_batched_tokens_from_extra_args():
    from trbench.eval.queue_report import batched_tokens
    assert batched_tokens({"extra_args": ["--max-num-batched-tokens", "4096", "--quantization", "gptq"]}) == 4096
    assert batched_tokens({"extra_args": ["--quantization", "gptq"]}) is None
    assert batched_tokens({}) is None                      # eski satirlar


def test_table_separates_batched_budgets():
    base = {"gpu_memory_utilization": 0.2, "prompt_tokens_target": None, "lang": "tr", "concurrency": 8,
            "repeat": 0, "tokens_per_s": 100, "ttft_p50_ms": 2000, "ttft_p95_ms": 2500, "ttft_p99_ms": 2600,
            "prompt_tokens_mean": 390, "queue_waiting_max": 0, "queue_waiting_mean": 0.0, "preemptions": 0}
    t = table([{**base, "batched_tokens": 2048}, {**base, "batched_tokens": 4096, "ttft_p50_ms": 600}])
    assert [(x["batched"], x["ttft50"]) for x in t] == [(2048, 2000), (4096, 600)]


def test_variant_from_row():
    from trbench.eval.queue_report import variant
    assert variant({}) == "cache on, metrics on"                       # eski 6d satirlari
    assert variant({"extra_args": ["--no-enable-prefix-caching"], "queue_metrics": False}) == "cache off, metrics off"
