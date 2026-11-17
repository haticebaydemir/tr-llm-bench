from trbench.eval.ifeval import (
    capitalize,
    check,
    has_upper,
    ifeval_items,
    keyword_count,
    score,
    tr_lower,
)


def test_turkish_lowercase_dotless_i():
    assert tr_lower("IRMAK İZMİR") == "ırmak izmir"   # Python .lower() 'irmak' derdi
    assert not has_upper("ırmak kıyısında", "tr")
    assert has_upper("Irmak", "tr")


def test_keyword_turkish_suffixes():
    text = "Deniz güzel. Denize girdim. Deniz'in rengi mavi. Dağ yüksek."
    assert keyword_count(text, "deniz", "tr") == 3


def test_keyword_english_plural_not_prefix():
    text = "The sea and seas. A season is not a sea."
    assert keyword_count(text, "sea", "en") == 3    # 'season' sayilmaz


def test_bullets_and_paragraphs():
    assert check("bullets", "* a\n* b\n* c", "tr", n=3)
    assert not check("bullets", "* a\n* b", "tr", n=3)
    assert check("paragraphs", "bir\n***\niki\n***\nüç", "tr", n=3)


def test_end_with_and_title_and_commas():
    assert check("end_with", "Metin. Başka sorunuz var mı?", "tr")
    assert check("title", "<<Deniz>>\nmetin", "tr")
    assert not check("no_commas", "a, b", "en")


def test_items_parallel_and_balanced():
    items = ifeval_items(60)
    tr = [i for i in items if i.lang == "tr"]
    en = [i for i in items if i.lang == "en"]
    assert len(tr) == len(en) == 60
    assert [i.id for i in tr] == [i.id for i in en]
    assert [len(i.meta["constraints"]) for i in tr] == [len(i.meta["constraints"]) for i in en]
    assert all(i.prompt[0].isupper() for i in items)       # cumle buyuk harfle baslar


def test_score_requires_all_constraints():
    item = next(i for i in ifeval_items(60)
                if i.lang == "tr" and [c["kind"] for c in i.meta["constraints"]]
                == ["no_commas", "max_words"])
    assert score(item, "kısa bir metin")
    assert not score(item, "kısa, bir metin")


def test_capitalize_turkish():
    assert capitalize("istanbul", "tr") == "İstanbul"   # Python 'Istanbul' derdi
    assert capitalize("ılgaz", "tr") == "Ilgaz"
