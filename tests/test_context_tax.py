from trbench.eval import context_tax as ct


def test_kv_bytes_per_token():
    # qwen2.5-7b: 28 katman x 4 kv x 128 boyut x 2 (K,V) x 2 bayt
    assert ct.kv_bytes_per_token("qwen2.5-7b") == 28 * 4 * 128 * 2 * 2


def test_depths_scale_with_tax():
    d = ct.depths("llama-3.1-8b", 1.5, 4.9)
    assert 1024 in d and 1536 in d and 4096 in d and 6144 in d


def test_depths_skip_over_budget():
    d = ct.depths("gemma-2-9b", 1.4, 5.8)
    assert max(d) < 4096


def test_table_ratio():
    rows = [
        {"model": "llama-3.1-8b", "depth": 4096, "n_prompt": 0, "n_gen": 64, "ts": 50.0},
        {"model": "llama-3.1-8b", "depth": 6144, "n_prompt": 0, "n_gen": 64, "ts": 40.0},
    ]
    out = ct.table(rows, {"llama-3.1-8b": 1.5}, e=4096)
    assert len(out) == 1 and out[0]["tr_tokens"] == 6144
