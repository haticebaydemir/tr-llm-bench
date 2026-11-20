import json

import pandas as pd

from trbench.eval.gpu_report import (
    engine_table,
    gguf_model,
    length_table,
    model_table,
    serving,
    top_conc,
)


def _row(engine, model, lang, c, rep, tps, ttft=100, peak=9000):
    return {"engine": engine, "model": model, "lang": lang, "concurrency": c, "repeat": rep,
            "status": "ok", "tokens_per_s": tps, "ttft_p50_ms": ttft, "errors": 0,
            "gpu2_peak_mib": peak, "gpu2_idle_mib": 3664}


def _write(p, rows):
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_serving_median_and_content(tmp_path):
    rows = [_row("llamacpp", "m", lang, c, r, tps)
            for lang in ("tr", "en") for c in (1, 8) for r, tps in enumerate((10 * c, 12 * c, 100 * c))]
    s = serving([_write(tmp_path / "a.jsonl", rows)])
    assert s.loc[(s.lang == "tr") & (s.concurrency == 8), "tps"].item() == 96   # medyan, ortalama degil
    assert set(top_conc(s)["concurrency"]) == {8}
    tok = pd.DataFrame({"tax": [1.6], "fert": [2.0]}, index=["m"])
    t = model_table(s, tok, usd_h=3.6)
    assert round(t.loc["m", "content"], 1) == 60.0                 # 96 / 1.6
    assert round(t.loc["m", "usd_per_1k_words"], 5) == round(1 / 48 * 1000 / 1000, 5)


def test_engine_table_relative_speed(tmp_path):
    rows = [_row("llamacpp", "mistral-7b-v0.3", "tr", 8, 0, 100), _row("vllm", "mistral-7b-v0.3", "tr", 8, 0, 150)]
    e = engine_table(serving([_write(tmp_path / "e.jsonl", rows)]))
    assert round(e["vllm_vs_llamacpp"].item(), 2) == 0.5


def test_length_uses_median_of_samples(tmp_path):
    rows = []
    for depth, samples in ((0, [37.4, 37.2, 21.5]), (4096, [33.6, 33.5, 33.5])):
        rows.append({"model_filename": "/g/Qwen__Qwen2.5-7B-Instruct-GGUF/x.gguf", "n_prompt": 0, "n_gen": 128,
                     "n_depth": depth, "samples_ts": samples})
        rows.append({"model_filename": "/g/Qwen__Qwen2.5-7B-Instruct-GGUF/x.gguf", "n_prompt": 4096, "n_gen": 0,
                     "n_depth": depth, "samples_ts": [700, 700, 700]})
    t = length_table(_write(tmp_path / "l.jsonl", rows))
    assert t.loc["qwen2.5-7b", "tg_0"] == 37.2                      # 21.5'lik tek dusus sonucu bozmaz
    assert gguf_model("/g/cosmos-tr-llama-8b/c.gguf") == "cosmos-tr-llama-8b"   # 'llama' ile karismaz
