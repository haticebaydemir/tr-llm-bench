from trbench.eval.analysis import TRANSLATION_ERRORS
from trbench.eval.export_hf import ERRATA, OWN, split


def test_third_party_text_never_exported():
    rows = [
        {"id": "json/1", "category": "json", "lang": "tr", "prompt": "p", "answer": "a",
         "scorer": "json", "source": OWN, "meta": {}},
        {"id": "math/x", "category": "math", "lang": "tr", "prompt": "GIZLI METIN", "answer": "5",
         "scorer": "number", "source": "malhajar/gsm8k_tr-v0.2", "meta": {}},
    ]
    own, ref = split(rows)
    assert [r["id"] for r in own] == ["json/1"]
    assert ref == [{"id": "math/x", "category": "math", "lang": "tr",
                    "source": "malhajar/gsm8k_tr-v0.2"}]
    assert "GIZLI" not in str(ref)


def test_errata_matches_analysis_list():
    assert set(ERRATA) == set(TRANSLATION_ERRORS)
