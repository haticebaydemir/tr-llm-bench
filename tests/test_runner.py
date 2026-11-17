import json

from trbench.config import ModelSpec
from trbench.runner import done_keys, prompts_for, run_key, served_name


def test_done_keys_only_counts_successful(tmp_path):
    p = tmp_path / "r.jsonl"
    rows = [
        {"model": "q", "quant": "fp16", "lang": "tr", "concurrency": 1, "repeat": 0, "status": "ok"},
        {"model": "q", "quant": "fp16", "lang": "tr", "concurrency": 4, "repeat": 0,
         "status": "engine_failed"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert done_keys(p) == {("q", "fp16", "tr", 1, 0)}      # basarisiz olan tekrar denenir
    assert done_keys(tmp_path / "yok.jsonl") == set()


def test_prompts_cycle_and_filter_by_lang():
    items = [{"lang": "tr", "prompt": "a"}, {"lang": "en", "prompt": "x"},
             {"lang": "tr", "prompt": "b"}]
    assert prompts_for(items, "tr", 5) == ["a", "b", "a", "b", "a"]


def test_run_key_and_served_name():
    assert run_key({"model": "m", "quant": "fp16", "lang": "en", "concurrency": 4,
                    "repeat": 2, "extra": 1}) == ("m", "fp16", "en", 4, 2)
    m = ModelSpec(name="qwen2.5-7b", quant="awq-int4", params_b=7.6, repo="x")
    assert served_name(m) == "trbench-qwen2.5-7b-awq-int4"
