import json

from trbench.eval.pareto import frontier, speeds, tr_accuracy


def test_speeds_reads_generation_rows_only():
    lines = [json.dumps({"model_filename": "/x/gemma-2-9b/gemma-2-9b-q8_0.gguf", "n_prompt": 512, "n_gen": 0, "avg_ts": 90.0}),
             json.dumps({"model_filename": "/x/gemma-2-9b/gemma-2-9b-q8_0.gguf", "n_prompt": 0, "n_gen": 128, "avg_ts": 6.5})]
    assert speeds(lines) == {"gemma-2-9b/q8_0": 6.5}


def test_frontier_drops_dominated_points():
    pts = [("a", 10, 50), ("b", 5, 80), ("c", 4, 60), ("d", 10, 40)]
    assert frontier(pts) == ["a", "b"]


def test_tr_accuracy_ignores_english_and_unpaired():
    rows = [{"label": "m/fp16", "lang": "tr", "id": "math/1", "correct": True},
            {"label": "m/fp16", "lang": "tr", "id": "math/2", "correct": False},
            {"label": "m/fp16", "lang": "en", "id": "math/1", "correct": False},
            {"label": "m/fp16", "lang": "tr", "id": "morphology/1", "correct": False}]
    assert tr_accuracy(rows) == {"m/fp16": 0.5}


def test_speeds_maps_official_qwen_split_files():
    line = json.dumps({"model_filename": "/g/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf",
                       "n_prompt": 0, "n_gen": 128, "avg_ts": 7.0})
    assert speeds([line]) == {"qwen2.5-7b/q8_0": 7.0}
