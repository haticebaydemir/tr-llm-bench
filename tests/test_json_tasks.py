from trbench.eval.json_tasks import extract_json, json_items, locative, score, tr_decimal


def test_locative_vowel_harmony_and_hard_consonants():
    assert locative("İzmir") == "İzmir'de"
    assert locative("Ankara") == "Ankara'da"
    assert locative("Kars") == "Kars'ta"      # s sert -> t
    assert locative("Muş") == "Muş'ta"
    assert locative("Erzurum") == "Erzurum'da"
    assert locative("Rize") == "Rize'de"


def test_tr_decimal():
    assert tr_decimal(7.5) == "7,50"


def test_items_parallel_same_values():
    items = json_items(10)
    tr = {i.id: i for i in items if i.lang == "tr"}
    en = {i.id: i for i in items if i.lang == "en"}
    assert tr.keys() == en.keys()
    p = tr["json/000"]
    assert "'" in p.prompt.split("Metin:")[1]            # TR metinde ekli isim var


def test_score_accepts_fenced_json_and_rejects_suffix():
    item = json_items(1)[0]                              # person, TR
    import json
    want = json.loads(item.answer)
    good = f"```json\n{json.dumps(want, ensure_ascii=False)}\n```"
    assert score(item, good)
    bad = dict(want, city=locative(want["city"]))        # "İzmir'de" -> yanlis
    assert not score(item, json.dumps(bad, ensure_ascii=False))
    assert not score(item, json.dumps(dict(want, age=str(want["age"]))))  # tip yanlis


def test_extract_json_garbage():
    assert extract_json("JSON yok") is None
    assert extract_json("{bozuk json") is None


def test_english_product_singular_ok():
    import json
    item = next(i for i in json_items(4) if i.lang == "en" and "product" in i.answer)
    want = json.loads(item.answer)
    assert score(item, json.dumps(dict(want, product=want["product"].rstrip("s"))))
