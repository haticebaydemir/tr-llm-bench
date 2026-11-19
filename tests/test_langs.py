import pandas as pd

from trbench.tokenizer_tax.fertility import sentence_counts
from trbench.tokenizer_tax.langs import ratios_with_ci, summary


def _counts(model, factors, n=200):
    rows = []
    for lang, f in factors.items():
        for i in range(n):
            base = 10 + (i % 7)                          # cumle zorlugu dillere ortak
            rows.append({"model": model, "lang": lang, "sent": i, "tokens": round(base * f)})
    return pd.DataFrame(rows)


def test_ratio_and_paired_ci():
    df = ratios_with_ci(_counts("m", {"eng_Latn": 1, "tur_Latn": 1.5, "fin_Latn": 1.5}), b=300)
    tr = df.set_index("lang").loc["tur_Latn"]
    assert abs(tr.ratio - 1.5) < 0.02 and tr.lo <= tr.ratio <= tr.hi
    assert tr.hi - tr.lo < 0.02          # eslestirilmis: oran sabitse aralik dar


def test_summary_classifies_turkish_against_agglutinative():
    c = pd.concat([_counts("a", {"eng_Latn": 1, "tur_Latn": 1.2, "fin_Latn": 1.8, "hun_Latn": 1.9}),
                   _counts("b", {"eng_Latn": 1, "tur_Latn": 1.6, "fin_Latn": 1.6, "hun_Latn": 1.6})])
    s = summary(ratios_with_ci(c, b=300)).set_index("model")["tr_vs_fi_hu"]
    assert s["a"] == "lower" and s["b"] == "within"


def test_sentence_counts_long_format():
    df = sentence_counts({"eng_Latn": ["a b", "c"], "tur_Latn": ["x", "y z"]},
                         lambda s: [len(x.split()) for x in s])
    assert list(df.columns) == ["lang", "sent", "tokens"] and df["tokens"].tolist() == [2, 1, 1, 2]


def test_headline_counts_shared_tokenizers_once():
    from trbench.tokenizer_tax.langs import headline
    f = {"eng_Latn": 1, "tur_Latn": 1.2, "fin_Latn": 1.8, "hun_Latn": 1.9}
    c = pd.concat([_counts("Qwen2.5-7B", f), _counts("Qwen3-8B", f),
                   _counts("x", {"eng_Latn": 1, "tur_Latn": 2.2, "fin_Latn": 1.8, "hun_Latn": 1.8})])
    assert headline(ratios_with_ci(c, b=200)) == \
        "Turkish costs fewer tokens than Finnish and Hungarian in only 1 of 2 tokenizers"
