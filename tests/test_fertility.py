from trbench.tokenizer_tax.fertility import compute_stats


def fake_counter(sents):
    # sahte tokenizer: her 3 karakter = 1 token
    return [max(1, len(s) // 3) for s in sents]


def test_compute_stats_ratios():
    texts = {
        "eng_Latn": ["I went to Ankara."],      # 4 kelime, 17 karakter -> 5 token
        "tur_Latn": ["Ankara'ya gittim."],       # 2 kelime, 17 karakter -> 5 token
    }
    df = compute_stats(texts, fake_counter).set_index("lang")
    assert df.loc["eng_Latn", "tax_vs_eng"] == 1.0
    assert df.loc["tur_Latn", "tax_vs_eng"] == 1.0
    # ayni token sayisi ama Turkce kelime basina 2 kat token: fertility yaniltici
    assert df.loc["tur_Latn", "fertility"] == 2.5
    assert df.loc["eng_Latn", "fertility"] == 1.25


def test_missing_reference_gives_nan():
    df = compute_stats({"tur_Latn": ["merhaba"]}, fake_counter)
    assert df["tax_vs_eng"].isna().all()
