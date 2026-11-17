import pandas as pd

from trbench.tokenizer_tax.plot import tr_ratios


def test_merges_shared_tokenizers_and_sorts():
    df = pd.DataFrame({
        "model": ["Qwen2.5-7B", "Qwen3-8B", "Mistral-7B-v0.3", "Qwen2.5-7B"],
        "lang": ["tur_Latn", "tur_Latn", "tur_Latn", "eng_Latn"],
        "tax_vs_eng": [1.62, 1.62, 2.21, 1.0],
    })
    s = tr_ratios(df)
    assert list(s.index) == ["Qwen2.5 / Qwen3", "Mistral-7B-v0.3"]
    assert s.iloc[0] == 1.62
