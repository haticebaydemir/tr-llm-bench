from trbench.eval.audit_pairs import audit, numbers


def test_numbers_normalize_turkish_and_english_formats():
    assert numbers("1.250 lira ve 3,5 kg", "tr") == numbers("1,250 lira and 3.5 kg", "en") == {"1250", "3.5"}


def _item(i, lang, q):
    return {"id": i, "lang": lang, "category": "math", "prompt": f"talimat\n\nProblem: {q}"}


def test_audit_flags_mismatched_pair():
    items = [_item("m1", "tr", "Ali 3 elma ve 5 armut aldi"), _item("m1", "en", "Ali bought 3 apples and 5 pears"),
             _item("m2", "tr", "Ayse 12 kalem aldi"), _item("m2", "en", "Tom has 7 dogs")]
    r = {x["id"]: x for x in audit(items)}
    assert r["m1"]["jaccard"] == 1.0
    assert r["m2"]["jaccard"] == 0.0 and r["m2"]["only_tr"] == ["12"]
