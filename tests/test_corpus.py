import pandas as pd

from trbench.tokenizer_tax.corpus import align, find_files


def test_find_files():
    files = [
        "dev/tur_Latn.parquet", "devtest/tur_Latn.parquet",
        "dev/eng_Latn.parquet", "README.md",
    ]
    assert find_files(files, "tur_Latn") == {
        "dev": "dev/tur_Latn.parquet", "devtest": "devtest/tur_Latn.parquet",
    }


def test_find_files_real_flores_layout():
    # 16 Kasim 2026'da depodaki gercek yapi: jsonl, 'ben_Beng' icinde 'eng' geciyor
    files = [
        "dev/ben_Beng.jsonl", "dev/eng_Latn.jsonl", "devtest/ben_Beng.jsonl",
        "devtest/eng_Latn.jsonl", "dataset_cards/eng_Latn.md",
    ]
    assert find_files(files, "eng_Latn") == {
        "dev": "dev/eng_Latn.jsonl", "devtest": "devtest/eng_Latn.jsonl",
    }


def test_align_inner_join_and_order():
    tr = pd.DataFrame({"id": ["b", "a", "c"], "text": ["iki", "bir", "uc"]})
    en = pd.DataFrame({"id": ["a", "b"], "text": ["one", "two"]})
    out = align({"tur_Latn": tr, "eng_Latn": en})
    assert list(out.columns) == ["id", "tur_Latn", "eng_Latn"]
    assert out["id"].tolist() == ["a", "b"]  # 'c' Ingilizcede yok -> atildi
    assert out.loc[0, "eng_Latn"] == "one"
