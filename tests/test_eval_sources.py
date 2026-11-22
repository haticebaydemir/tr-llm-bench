import pandas as pd
import pytest

from trbench.eval.schema import Item
from trbench.eval.sources import (
    belebele_items,
    final_number,
    global_mmlu_items,
    gsm8k_items,
    pair_gsm8k,
    pick_parallel_ids,
)


def fake_belebele(lang_word: str) -> pd.DataFrame:
    return pd.DataFrame({
        "link": ["a", "a", "b", "c"],
        "question_number": [1, 2, 1, 1],
        "flores_passage": [f"{lang_word} metin"] * 4,
        "question": ["q1", "q2", "q3", "q4"],
        "mc_answer1": ["x"] * 4, "mc_answer2": ["y"] * 4,
        "mc_answer3": ["z"] * 4, "mc_answer4": ["w"] * 4,
        "correct_answer_num": [1, 2, 3, 4],
    })


def test_parallel_ids_are_deterministic_and_shared():
    tr, en = fake_belebele("tr"), fake_belebele("en").iloc[:3]  # 'c#1' sadece TR'de
    a = pick_parallel_ids(tr, en, 2)
    assert a == pick_parallel_ids(tr, en, 2)
    assert "c#1" not in a


def test_belebele_items_parallel_and_answer_letter():
    tr, en = fake_belebele("tr"), fake_belebele("en")
    ids = ["a#2", "c#1"]
    t, e = belebele_items(tr, "tr", ids), belebele_items(en, "en", ids)
    assert [i.id for i in t] == [i.id for i in e]          # paralel
    assert [i.answer for i in t] == ["B", "D"]             # 1-indeksli -> harf
    assert "Cevap:" in t[0].prompt and "Answer:" in e[0].prompt


def test_item_rejects_bad_category():
    with pytest.raises(ValueError):
        Item(id="x", category="summary", lang="tr", prompt="", answer="", scorer="", source="")


def test_final_number_formats():
    assert final_number("... #### 42") == "42"
    assert final_number("... #### 1,250") == "1250"      # EN binlik
    assert final_number("... #### 1.250") == "1250"      # TR binlik
    assert final_number("... #### 2.5") == "2.5"          # ondalik korunur
    assert final_number("cevap yok") is None


def _df(nums):
    return pd.DataFrame({"question": [f"q{n}" for n in nums],
                         "answer": [f"#### {n}" for n in nums]})


def test_pairing_survives_deleted_row():
    # Gercek durum: TR'de bir satir eksik. Sira-sira eslestirme 5'ten sonrasini bozardi.
    en = _df([10, 11, 12, 13, 14, 15, 16, 17, 18])
    tr = _df([10, 11, 12, 13, 15, 16, 17, 18])            # 14 silinmis
    pairs = pair_gsm8k(tr, en)
    assert (4, 5) in pairs                                  # tr[4]=15 <-> en[5]=15
    assert all(tr["answer"][i] == en["answer"][j] for i, j in pairs)
    assert len(pairs) == 8


def test_pairing_ignores_short_coincidences():
    en = _df([1, 2, 3, 4, 5, 6])
    tr = _df([1, 2, 3, 99, 5, 98])                          # tek eslesen '5' tesaduf olabilir
    assert pair_gsm8k(tr, en) == [(0, 0), (1, 1), (2, 2)]


def test_gsm8k_items_parallel():
    en = _df(range(5))
    tr = en.assign(question=[f"s{i}" for i in range(5)])
    items = gsm8k_items(tr, en, 3)
    t = [i for i in items if i.lang == "tr"]
    e = [i for i in items if i.lang == "en"]
    assert [i.id for i in t] == [i.id for i in e]
    assert all(a.answer == b.answer for a, b in zip(t, e))


def _mmlu(lang, labels):
    n = len(labels)
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "question": [f"{lang} soru {i}" for i in range(n)],
        "option_a": ["a"] * n, "option_b": ["b"] * n,
        "option_c": ["c"] * n, "option_d": ["d"] * n,
        "answer": ["c"] * n,
        "cultural_sensitivity_label": labels,
        "subject": ["physics"] * n,
    })


def test_global_mmlu_only_culturally_agnostic():
    labels = ["CA", "CS", "CA", "CS", "CA"]
    items = global_mmlu_items(_mmlu("tr", labels), _mmlu("en", labels), 3)
    ids = {i.id for i in items}
    assert ids == {"knowledge/s0", "knowledge/s2", "knowledge/s4"}   # CS'ler disarida
    assert all(i.answer == "C" for i in items)                      # kucuk harf -> buyuk


def test_global_mmlu_reports_missing_columns():
    bad = _mmlu("tr", ["CA"]).drop(columns=["option_d"])
    with pytest.raises(ValueError, match="option_d"):
        global_mmlu_items(bad, _mmlu("en", ["CA"]), 1)


def test_global_mmlu_requires_cultural_label():
    # Sutun yoksa CA filtresi yapilamaz: sessizce butun sorulari almak yerine hata
    bad = _mmlu("tr", ["CA"]).drop(columns=["cultural_sensitivity_label"])
    with pytest.raises(ValueError, match="cultural_sensitivity_label"):
        global_mmlu_items(bad, _mmlu("en", ["CA"]), 1)


def test_math_prompt_requires_reasoning():
    # Regresyon: "dusunebilirsiniz" (izin) degil, "cozun" (zorunlu) olmali
    from trbench.eval.sources import MATH_PROMPT
    assert "adım adım çözün" in MATH_PROMPT["tr"] and "yalnızca" not in MATH_PROMPT["tr"]
    assert "step by step first" in MATH_PROMPT["en"] and "only" not in MATH_PROMPT["en"]


def test_gsm8k_verified_pairs_used_and_checked(tmp_path):
    from trbench.eval.sources import choose_gsm8k, load_verified_pairs, save_verified_pairs
    en = _df(range(6))
    tr = en.assign(question=[f"s{i}" for i in range(6)])
    chosen = choose_gsm8k(tr, en, 3)
    f = tmp_path / "p.json"
    save_verified_pairs(chosen, tr, en, f)
    v = load_verified_pairs(f)
    assert v == chosen and choose_gsm8k(tr, en, 3, v) == chosen
    assert [i.id for i in gsm8k_items(tr, en, 3, v)] == [i.id for i in gsm8k_items(tr, en, 3)]
    assert load_verified_pairs(tmp_path / "yok.json") is None
    with pytest.raises(ValueError, match="uyusmuyor"):
        choose_gsm8k(tr, en, 3, [(0, 1), (1, 1), (2, 2)])           # cevaplari farkli cift


def test_global_mmlu_rejects_duplicate_sample_ids():
    df = _mmlu("tr", ["CA", "CA", "CA"])
    df.loc[1, "sample_id"] = "s0"
    with pytest.raises(ValueError, match="tekrarlanan"):
        global_mmlu_items(df, _mmlu("en", ["CA", "CA", "CA"]), 1)


def test_global_mmlu_selects_from_filtered_table():
    # Ayni sample_id hem CS hem CA olarak varsa (tekillik bozuksa) CA satiri alinmali
    tr = _mmlu("tr", ["CS", "CA", "CA"])
    tr.loc[0, "sample_id"] = "s1"                       # CS kopya once geliyor
    tr.loc[0, "question"] = "YANLIS"
    items = global_mmlu_items(tr, _mmlu("en", ["CS", "CA", "CA"]), 2)
    assert all("YANLIS" not in i.prompt for i in items)
