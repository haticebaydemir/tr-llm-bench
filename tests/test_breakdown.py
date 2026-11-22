import math

from trbench.eval.breakdown import cell, gap_ci, paired, wilson


def test_wilson_bounds():
    lo, hi = wilson(8, 10)
    assert 0.49 < lo < 0.8 < hi < 0.95
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_gap_ci_paired():
    g, lo, hi = gap_ci([(True, False)] * 10)
    assert g == lo == hi == 1.0
    g, lo, hi = gap_ci([(True, True), (False, False)] * 20)
    assert g == lo == hi == 0.0


def test_cell_groups_and_excludes_translation_errors():
    rows = [{"label": "m/q", "id": f"{c}/{i}", "lang": l, "correct": ok}
            for c, ok_tr in (("reading", True), ("json", False))
            for i in range(4) for l, ok in (("tr", ok_tr), ("en", True))]
    rows += [{"label": "m/q", "id": "math/gsm8k-test-721", "lang": "tr", "correct": False},
             {"label": "m/q", "id": "math/gsm8k-test-721", "lang": "en", "correct": True}]
    qs = paired(rows)["m/q"]
    pub = cell(qs, ("reading", "knowledge", "math"))
    org = cell(qs, ("instruction", "json"))
    assert pub["n"] == 4 and pub["tr"] == 1.0              # ceviri hatali GSM8K sorusu sayilmadi
    assert org["n"] == 4 and org["tr"] == 0.0 and org["gap"] == -1.0
