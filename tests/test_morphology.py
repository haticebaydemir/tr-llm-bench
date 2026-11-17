from trbench.eval.morphology import morphology_items, score, suffix


def test_cases_vowel_harmony_buffer_and_hard_consonant():
    assert suffix("Ankara", "dative") == "Ankara'ya"       # unluden sonra kaynastirma y
    assert suffix("İzmir", "dative") == "İzmir'e"
    assert suffix("Kars", "locative") == "Kars'ta"          # sert unsuz
    assert suffix("Rize", "ablative") == "Rize'den"
    assert suffix("Muş", "ablative") == "Muş'tan"
    assert suffix("Erzurum", "genitive") == "Erzurum'un"    # 4'lu uyum: u
    assert suffix("Rize", "genitive") == "Rize'nin"         # unluden sonra n
    assert suffix("Van", "genitive") == "Van'ın"
    assert suffix("Sinop", "dative") == "Sinop'a"           # ozel isim: yumusama yok


def test_items_tr_only_and_count():
    items = morphology_items(40)
    assert len(items) == 50
    assert {i.lang for i in items} == {"tr"}


def test_score_lenient_format_strict_form():
    item = morphology_items(40)[0]
    assert score(item, item.answer)
    assert score(item, f"Cevap: {item.answer.replace(chr(39), '’')}.")
    assert not score(item, item.answer.split("'")[0])        # eksiz cevap yanlis
