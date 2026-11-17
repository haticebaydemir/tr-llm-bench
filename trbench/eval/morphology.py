"""Turkce morfoloji: ek uretimi. SADECE TR (Ingilizce karsiligi yok) ->
TR/EN kiyasinin disinda, 'quantization Turkce dilbilgisini bozuyor mu?' sorusu icin.

Ozel isimler secildi: yazimda unsuz yumusamasi olmaz (Sinop'a, Zonguldak'a),
boylece kural istisnasiz ve dogru cevap tek.
"""

from __future__ import annotations

import random
import re

from trbench.eval.json_tasks import CITIES, HARD
from trbench.eval.schema import Item

SEED = 2026
VOWELS = "aeıioöuü"


def _last_vowel(w: str) -> str:
    return next((c for c in reversed(w.lower()) if c in VOWELS), "a")


def _two_way(w: str) -> str:          # a / e
    return "e" if _last_vowel(w) in "eiöü" else "a"


def _four_way(w: str) -> str:         # ı / i / u / ü
    return {"a": "ı", "ı": "ı", "e": "i", "i": "i", "o": "u", "u": "u",
            "ö": "ü", "ü": "ü"}[_last_vowel(w)]


def suffix(name: str, case: str) -> str:
    """Ozel isme hal eki: yonelme, bulunma, ayrilma, ilgi."""
    ends_vowel = name[-1].lower() in VOWELS
    hard = name[-1].lower() in HARD
    a, i = _two_way(name), _four_way(name)
    if case == "dative":
        return f"{name}'{'y' if ends_vowel else ''}{a}"
    if case == "locative":
        return f"{name}'{'t' if hard else 'd'}{a}"
    if case == "ablative":
        return f"{name}'{'t' if hard else 'd'}{a}n"
    if case == "genitive":
        return f"{name}'{'n' if ends_vowel else ''}{i}n"
    raise ValueError(case)


CASE_NAME = {"dative": "yönelme", "locative": "bulunma", "ablative": "ayrılma",
             "genitive": "ilgi"}
PLURALS = [("kitap", "kitaplar"), ("ev", "evler"), ("göz", "gözler"), ("kapı", "kapılar"),
           ("çiçek", "çiçekler"), ("kuş", "kuşlar"), ("köprü", "köprüler"),
           ("masa", "masalar"), ("okul", "okullar"), ("gemi", "gemiler")]


def morphology_items(n_case: int = 40) -> list[Item]:
    rng = random.Random(SEED)
    pool = [(c, k) for c in CITIES for k in CASE_NAME]
    items = []
    for idx, (city, case) in enumerate(sorted(rng.sample(pool, n_case))):
        ex = suffix("İstanbul", case)
        items.append(Item(
            id=f"morphology/case-{idx:03d}", category="morphology", lang="tr",
            prompt=(f"'{city}' özel ismine {CASE_NAME[case]} hâli ekini getir. "
                    f"Örnek: İstanbul → {ex}. Yalnızca sonucu yaz."),
            answer=suffix(city, case), scorer="morph", source="trbench (ozgun sablon)",
            meta={"rule": case},
        ))
    for idx, (word, plural) in enumerate(PLURALS):
        items.append(Item(
            id=f"morphology/plural-{idx:03d}", category="morphology", lang="tr",
            prompt=f"'{word}' kelimesinin çoğulunu yaz. Yalnızca sonucu yaz.",
            answer=plural, scorer="morph", source="trbench (ozgun sablon)",
            meta={"rule": "plural"},
        ))
    return items


def score(item: Item, response: str) -> bool:
    """Cevap kisa olmali ve beklenen bicim birebir gecmeli (kesme isareti turu esnek)."""
    norm = response.replace("’", "'")
    tokens = re.findall(r"\w+(?:'\w+)*", norm)
    return 0 < len(tokens) <= 5 and item.answer in tokens
