import json

from trbench.config import ModelSpec, load
from trbench.runner import done_keys, llamacpp_cmd, prompts_for, run_key, served_name


def test_done_keys_only_counts_successful(tmp_path):
    p = tmp_path / "r.jsonl"
    rows = [
        {"model": "q", "quant": "fp16", "lang": "tr", "concurrency": 1, "repeat": 0, "status": "ok"},
        {"model": "q", "quant": "fp16", "lang": "tr", "concurrency": 4, "repeat": 0,
         "status": "engine_failed"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert done_keys(p) == {("vllm", "q", "fp16", "tr", 1, 0)}   # eski satirlar: vllm; basarisiz tekrar denenir
    assert done_keys(tmp_path / "yok.jsonl") == set()


def test_prompts_cycle_and_filter_by_lang():
    items = [{"lang": "tr", "prompt": "a"}, {"lang": "en", "prompt": "x"},
             {"lang": "tr", "prompt": "b"}]
    assert prompts_for(items, "tr", 5) == ["a", "b", "a", "b", "a"]


def test_run_key_and_served_name():
    assert run_key({"engine": "llamacpp", "model": "m", "quant": "fp16", "lang": "en",
                    "concurrency": 4, "repeat": 2, "extra": 1}) == ("llamacpp", "m", "fp16", "en", 4, 2)
    m = ModelSpec(name="qwen2.5-7b", quant="awq-int4", params_b=7.6, repo="x")
    assert served_name(m) == "trbench-qwen2.5-7b-awq-int4"


def test_llamacpp_cmd_full_offload_slots_and_template():
    from pathlib import Path
    cfg = load("configs/gpu_llamacpp.yaml")
    t = next(m for m in cfg.models if m.name == "trendyol-7b")
    cmd = llamacpp_cmd(cfg, t, Path("x.gguf"))
    assert cmd[cmd.index("-ngl") + 1] == "99"
    assert cmd[cmd.index("-np") + 1] == "8" and cmd[cmd.index("-c") + 1] == str(1536 * 8)
    assert "--chat-template-file" in cmd and "trbench-trendyol-7b-q4_k_m" in cmd
    g = next(m for m in cfg.models if m.name == "gemma-2-9b")
    assert llamacpp_cmd(cfg, g, Path("g.gguf"))[llamacpp_cmd(cfg, g, Path("g.gguf")).index("-np") + 1] == "2"
