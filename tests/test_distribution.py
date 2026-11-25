import pandas as pd

from trbench.tokenizer_tax.distribution import analyse, identical_groups, wide

LANGS = ("eng_Latn", "tur_Latn", "fin_Latn", "hun_Latn")


def _counts(model, mult, n=20):
    rows = []
    for s in range(n):
        en = 10 + s
        for lang, k in zip(LANGS, (1.0, mult, 2.0, 2.0), strict=True):
            rows.append({"lang": lang, "sent": s, "tokens": round(en * k), "model": model})
    return rows


def test_identical_tokenizers_merged():
    df = pd.DataFrame(_counts("a", 1.5) + _counts("b", 1.5) + _counts("c", 0.5))
    groups = identical_groups(wide(df))
    assert sorted(map(sorted, groups)) == [["a", "b"], ["c"]]


def test_analyse_ratios():
    df = pd.DataFrame(_counts("a", 1.5) + _counts("c", 0.5))
    res = analyse(df, {"a": 100, "c": 200})
    by = {r["tokenizer"]: r for r in res["rows"]}
    assert res["n_independent"] == 2
    assert abs(by["c"]["tr_vs_fi_hu"] - 0.25) < 0.02
    assert by["c"]["tr_cheaper_share"] == 1.0 and by["a"]["tr_cheaper_share"] == 0.0
    assert [r["tokenizer"] for r in res["rows"]] == ["c", "a"]
