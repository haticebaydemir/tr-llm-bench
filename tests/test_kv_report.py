from trbench.eval.kv_report import report, tables


def _rows():
    rows = []
    for i in range(20):
        for lang in ("tr", "en"):
            for suf, ok in (("kvf16", True), ("kvq8", True), ("kvq4", lang == "en" or i % 2 == 0)):
                rows.append({"label": f"m-1b-{suf}/q4_k_m", "id": f"math/q{i}", "lang": lang,
                             "category": "math", "correct": ok})
    return rows


def test_tables_split_label():
    t = tables(_rows())
    assert set(t) == {"m-1b"} and set(t["m-1b"]) == {"fp16", "q8_0", "q4_0"}


def test_report_tax():
    md = report(_rows())
    assert "| m-1b | q8_0 | +0.0 | +0.0 | +0.0" in md
    assert "| m-1b | q4_0 | +50.0 | +0.0 | +50.0" in md
