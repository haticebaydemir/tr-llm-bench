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
