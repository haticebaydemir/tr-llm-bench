import pytest

from trbench.eval.schema import Item
from trbench.eval.scorers import parse_choice, parse_number, score, to_number


@pytest.mark.parametrize("resp,want", [
    ("B", "B"), ("Cevap: C", "C"), ("**D**", "D"), ("(A) çünkü...", "A"),
    ("Answer: b", "B"), ("Doğru seçenek B) şıkkıdır.", "B"),
    ("A good question. The answer is C)", "C"),     # baştaki 'A' artikeli sayilmaz
    ("B şıkkı doğru", "B"), ("c", "C"),
    ("Bilmiyorum", None),
])
def test_parse_choice(resp, want):
    assert parse_choice(resp) == want


def test_numbers_locale():
    assert to_number("1.250", "tr") == 1250
    assert to_number("3,75", "tr") == 3.75
    assert to_number("1,250", "en") == 1250
    assert to_number("3.75", "en") == 3.75


def test_parse_number_prefers_label_then_last():
    assert parse_number("16 gözlük, 8'i indirimli... Cevap: 64", "tr") == 64
    assert parse_number("First 5, then 3, total 64.", "en") == 64
    assert parse_number("Answer: $1,250", "en") == 1250


def _item(scorer, answer, lang="tr"):
    return Item(id="x", category="math", lang=lang, prompt="", answer=answer,
                scorer=scorer, source="")


def test_dispatch():
    assert score(_item("number", "64"), "...\nCevap: 64")
    assert not score(_item("number", "64"), "Cevap: 46")
    assert score(_item("choice", "B"), "Cevap: B")
    with pytest.raises(ValueError):
        score(_item("rouge", "x"), "x")


def test_parse_choice_letter_then_newline_explanation():
    from trbench.eval.scorers import parse_choice
    assert parse_choice('A\n\nMetinde belirtildiği gibi, bir deneyin amacı...') == "A"
    assert parse_choice("D\nBecause the text says so.") == "D"
    assert parse_choice("A good answer is C) because") == "C"   # 'A' artikeli hala sayilmaz


def test_reasoning_block_is_not_scored():
    from trbench.eval.scorers import strip_reasoning
    assert strip_reasoning("<think>\nA mi B mi?\n</think>\n\nCevap: C") == "Cevap: C"
    assert strip_reasoning("<think>bitmedi") == ""
    assert strip_reasoning("Cevap: B") == "Cevap: B"
