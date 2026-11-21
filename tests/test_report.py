from trbench.eval.report import board, write_space


def _rows(label, tr, en):
    rows = []
    for i in range(10):
        rows.append({"label": label, "lang": "tr", "id": f"math/{i}", "correct": i < tr, "category": "math"})
        rows.append({"label": label, "lang": "en", "id": f"math/{i}", "correct": i < en, "category": "math"})
    return rows


def test_board_computes_accuracy_and_content_speed(tmp_path, monkeypatch):
    rows = _rows("qwen2.5-7b/fp16", 7, 9)
    b = board(rows, {"qwen2.5-7b/fp16": 3.24})
    assert b == [{"model": "qwen2.5-7b", "quant": "fp16", "tr": 70.0, "en": 90.0, "morph": None,
                  "tok_tax": 1.62, "cpu_tps": 3.2, "tr_content_tps": 2.0}]
    monkeypatch.chdir(tmp_path)
    write_space(b, "2026-11-18")
    page = (tmp_path / "space" / "index.html").read_text(encoding="utf-8")
    assert "qwen2.5-7b" in page and "-20.0" in page and "sdk: static" in (tmp_path / "space" / "README.md").read_text()


def test_gpu_board_accuracy_ci_categories_and_content_speed():
    from trbench.eval.report import gpu_board
    q = []
    for i in range(20):
        for lang in ("tr", "en"):
            q.append({"label": "qwen3-8b/q4_k_m", "id": f"math/{i}", "lang": lang, "category": "math",
                      "correct": i < (15 if lang == "tr" else 18)})
    q.append({"label": "qwen3-8b/q4_k_m", "id": "morphology/1", "lang": "tr", "category": "morphology", "correct": True})
    s = [{"model": "qwen3-8b", "lang": "tr", "concurrency": c, "status": "ok", "tokens_per_s": v}
         for c, v in ((1, 30), (8, 120), (8, 130), (8, 500))]
    g = gpu_board(q, s, {"qwen3-8b": 1.6})
    r = g[0]
    assert (r["tr"], r["en"], r["math"], r["morph"]) == (75.0, 90.0, 75.0, 100.0)
    assert r["tr_ci"] == 19.0                     # 1.96*sqrt(.75*.25/20)
    assert r["gpu_tps"] == 130 and r["tr_content_tps"] == 81     # medyan(120,130,500)=130 /1.6
