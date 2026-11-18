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
