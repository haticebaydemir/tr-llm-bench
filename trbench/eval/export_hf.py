"""Eval setini Hugging Face'e yayina hazirlar (yukleme ELLE, onayla yapilir).

Lisans karari (17 Kasim):
  - Ozgun sablonlarimiz (talimat, JSON, morfoloji) -> tam metin, CC-BY-4.0.
  - Ucuncu kisi kaynaklar (Belebele, Global-MMLU, GSM8K-TR) -> SADECE id + kaynak.
    GSM8K-TR v0.2'nin lisansi belirtilmemis; metni yeniden dagitmiyoruz.
    Tam set: python -m trbench.eval.build ile kaynaklardan yeniden kurulur.
  - 8 ceviri hatasi listesi (bizim incelememiz) -> errata.

Kullanim:  python -m trbench.eval.export_hf   -> data/hf/
"""

from __future__ import annotations

import json
from pathlib import Path

from trbench.eval.analysis import TRANSLATION_ERRORS

SRC = Path("data/eval/eval_set.jsonl")
OUT = Path("data/hf")
OWN = "trbench (ozgun sablon)"

ERRATA = {
    "reading/https://en.wikinews.org/wiki/Building_collapses_in_Mecca_at_eve_of_Hajj_pillgrimage#1":
        "Option D negated in translation: the false statement became true.",
    "reading/https://en.wikinews.org/wiki/Sun%27s_mood_swings_not_so_strange_after_all,_say_scientists#1":
        "'rotation' translated as 'yörünge' (orbit) in the passage but 'dönme' in the options.",
    "reading/https://en.wikivoyage.org/wiki/Aggregators#1":
        "'airline' translated as 'havaalanı' (airport) in the options.",
    "reading/https://en.wikivoyage.org/wiki/Noum%C3%A9a#2":
        "Passage and options use different Turkish terms for 'flat white' and 'short black'.",
    "reading/https://en.wikinews.org/wiki/The_Onion:_An_interview_with_%27America%27s_Finest_News_Source%27#2":
        "'empire' translated as 'programlar', making a distractor plausible (borderline).",
    "knowledge/high_school_biology/test/211":
        "'EXCEPT' dropped in translation, reversing the question.",
    "knowledge/college_medicine/test/72":
        "'H2O(s)' (solid) rendered as 'H2O(lar)', a Turkish plural suffix, in two options.",
    "math/gsm8k-test-721":
        "'200 taps per minute with each foot' became ambiguous ('her iki ayağıyla').",
}


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(tam metinle yayinlanan, sadece referansla yayinlanan)"""
    own = [r for r in rows if r["source"] == OWN]
    ref = [{"id": r["id"], "category": r["category"], "lang": r["lang"], "source": r["source"]}
           for r in rows if r["source"] != OWN]
    return own, ref


def main() -> None:
    assert set(ERRATA) == set(TRANSLATION_ERRORS), "errata ve analiz listesi ayni olmali"
    rows = [json.loads(x) for x in SRC.read_text(encoding="utf-8").splitlines()]
    own, ref = split(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("items.jsonl", own), ("references.jsonl", ref)):
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data),
                                encoding="utf-8")
    errata = [{"id": k, "issue": v} for k, v in sorted(ERRATA.items())]
    (OUT / "errata.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in errata), encoding="utf-8")
    print(f"items: {len(own)}  references: {len(ref)}  errata: {len(errata)}  -> {OUT}/")


if __name__ == "__main__":
    main()
