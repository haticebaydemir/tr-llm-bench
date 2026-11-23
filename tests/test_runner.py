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
    assert done_keys(p) == {("vllm", "q", "fp16", "tr", 1, 0, None, None)}   # eski satirlar: vllm; basarisiz tekrar denenir
    assert done_keys(tmp_path / "yok.jsonl") == set()


def test_prompts_cycle_and_filter_by_lang():
    items = [{"lang": "tr", "prompt": "a"}, {"lang": "en", "prompt": "x"},
             {"lang": "tr", "prompt": "b"}]
    assert prompts_for(items, "tr", 5) == ["a", "b", "a", "b", "a"]


def test_run_key_and_served_name():
    assert run_key({"engine": "llamacpp", "model": "m", "quant": "fp16", "lang": "en",
                    "concurrency": 4, "repeat": 2, "fp": "abc", "exp": "e1", "extra": 1}) == (
        "llamacpp", "m", "fp16", "en", 4, 2, "abc", "e1")
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


def test_vllm_extra_args_from_config(monkeypatch):
    from pathlib import Path

    import trbench.runner as R
    seen = {}
    monkeypatch.setattr(R.subprocess, "Popen", lambda cmd, **k: seen.setdefault("cmd", cmd))
    monkeypatch.setattr(R, "LOGS", Path("."))
    monkeypatch.delenv("VLLM_EXTRA", raising=False)
    cfg = load("configs/gpu_vllm.yaml")
    R.start_vllm(cfg, cfg.models[0], Path("m"))
    cmd = seen["cmd"]
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "2048" and "--enforce-eager" in cmd
    Path("vllm_trbench-mistral-7b-v0.3-awq-int4.log").unlink(missing_ok=True)


def test_fingerprint_changes_with_result_affecting_settings():
    # max_tokens degisirse eski kosu "bitmis" sayilmamali (resume yanlis atlamasin)
    cfg = load("configs/gpu_llamacpp.yaml")
    m = cfg.models[0]
    fp = cfg.fingerprint(m)
    assert all(r["fp"] == fp for r in cfg.runs() if r["model"] == m.name)
    assert cfg.model_copy(update={"max_tokens": 128}).fingerprint(m) != fp
    assert cfg.model_copy(update={"concurrency": [1]}).fingerprint(m) == fp   # kosu ekseni, ayar degil


def test_done_keys_depend_on_experiment_id(tmp_path):
    """Ayni kosu baska bir deney kimligiyle (eval seti, agirlik, kod...) olculduyse 'bitmemis' sayilir."""
    p = tmp_path / "r.jsonl"
    row = {"engine": "llamacpp", "model": "m", "quant": "q4_k_m", "lang": "tr", "concurrency": 1,
           "repeat": 0, "fp": "f", "exp": "eski", "status": "ok"}
    p.write_text(json.dumps(row), encoding="utf-8")
    assert run_key({**row, "exp": "eski"}) in done_keys(p)
    assert run_key({**row, "exp": "yeni"}) not in done_keys(p)


def test_token_windows_equal_size():
    from trbench.runner import token_windows
    w = token_windows(list(range(10)), 3, 5)
    assert [len(x) for x in w] == [3] * 5 and w[0] == [0, 1, 2] and w[3] == [0, 1, 2]


def test_new_config_fields_keep_old_fingerprint():
    cfg = load("configs/gpu_llamacpp.yaml")
    m = cfg.models[0]
    old = cfg.fingerprint(m)
    assert cfg.model_copy(update={"queue_metrics": False}).fingerprint(m) == old
    assert cfg.model_copy(update={"queue_metrics": True}).fingerprint(m) != old
    assert cfg.model_copy(update={"prompt_tokens": 128}).fingerprint(m) != old
    assert all(r["n_requests"] == max(16, 4 * r["concurrency"]) for r in cfg.runs())
