"""Yapilandirilmis cikti: metinden bilgi cikar -> JSON. TR/EN paralel, sablondan.

Turkce'ye ozgu iki tuzak bilerek konuldu:
  1) Ekli ozel isim: metinde "İzmir'de", JSON'da "İzmir" bekleniyor.
  2) Ondalik virgul: metinde "7,50 TL", JSON'da 7.5 sayisi bekleniyor.
"""

from __future__ import annotations

import json
import random
import re

from trbench.eval.schema import Item

SEED = 2026
NAMES = ["Ayşe", "Mehmet", "Elif", "Can", "Zeynep", "Burak", "Selin", "Emre", "Deniz", "Kerem"]
CITIES = ["İzmir", "Ankara", "Bursa", "Erzurum", "Trabzon", "Kars", "Antalya", "Eskişehir",
          "Muş", "Van", "Sinop", "Rize"]
PRODUCTS = [("kalem", "pens"), ("defter", "notebooks"), ("silgi", "erasers"),
            ("bardak", "glasses"), ("kitap", "books"), ("tabak", "plates")]

FRONT_VOWELS, BACK_VOWELS = set("eiöü"), set("aıou")
HARD = set("çfhkpsşt")  # "fıstıkçı şahap": bu harflerle biten kelimeye -da/-de degil -ta/-te


def locative(word: str) -> str:
    """Ozel isme bulunma eki: İzmir'de, Ankara'da, Kars'ta, Muş'ta."""
    last_vowel = next((c for c in reversed(word.lower()) if c in FRONT_VOWELS | BACK_VOWELS), "a")
    vowel = "e" if last_vowel in FRONT_VOWELS else "a"
    cons = "t" if word[-1].lower() in HARD else "d"
    return f"{word}'{cons}{vowel}"


def tr_decimal(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


PROMPT = {
    "tr": ("Aşağıdaki metinden bilgileri çıkar. Yalnızca şu şemaya uyan geçerli bir JSON "
           "döndür, başka hiçbir şey yazma:\n{schema}\n\nMetin: {text}"),
    "en": ("Extract the information from the text below. Return only valid JSON matching "
           "this schema and nothing else:\n{schema}\n\nText: {text}"),
}
SCHEMA = {
    "person": {"tr": '{"name": metin, "age": tam sayı, "city": metin}',
               "en": '{"name": string, "age": integer, "city": string}'},
    "order": {"tr": '{"product": metin, "quantity": tam sayı, "unit_price": sayı}',
              "en": '{"product": string, "quantity": integer, "unit_price": number}'},
}


def _person(rng: random.Random) -> tuple[dict, dict, dict]:
    name, city, age = rng.choice(NAMES), rng.choice(CITIES), rng.randint(19, 78)
    text = {"tr": f"{name} {age} yaşında ve {locative(city)} yaşıyor.",
            "en": f"{name} is {age} years old and lives in {city}."}
    return text, {"tr": {"name": name, "age": age, "city": city},
                  "en": {"name": name, "age": age, "city": city}}, SCHEMA["person"]


def _order(rng: random.Random) -> tuple[dict, dict, dict]:
    (tr_p, en_p), qty = rng.choice(PRODUCTS), rng.randint(2, 40)
    price = rng.choice([2.5, 3.75, 7.5, 12.25, 19.9, 4.6])
    text = {"tr": f"Mağaza {qty} adet {tr_p} sattı, tanesi {tr_decimal(price)} TL.",
            "en": f"The store sold {qty} {en_p} at {price:.2f} TL each."}
    return text, {"tr": {"product": tr_p, "quantity": qty, "unit_price": price},
                  "en": {"product": en_p, "quantity": qty, "unit_price": price}}, SCHEMA["order"]


def json_items(n: int = 50) -> list[Item]:
    rng = random.Random(SEED)
    items = []
    for idx in range(n):
        text, expected, schema = (_person if idx % 2 == 0 else _order)(rng)
        for lang in ("tr", "en"):
            items.append(Item(
                id=f"json/{idx:03d}",
                category="json",
                lang=lang,
                prompt=PROMPT[lang].format(schema=schema[lang], text=text[lang]),
                answer=json.dumps(expected[lang], ensure_ascii=False),
                scorer="json",
                source="trbench (ozgun sablon)",
            ))
    return items


def extract_json(response: str) -> dict | None:
    """```json ... ``` cercevesi veya oncesi/sonrasi metin olsa da ilk {...} blogunu ayiklar."""
    m = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _same_product(got: str, want: str, lang: str) -> bool:
    """EN'de 'books' yerine 'book' yazmak bilgi hatasi degil -> tekil/cogul esit sayilir."""
    g, w = got.strip().lower(), want.lower()
    if lang == "en":
        return g.rstrip("s") == w.rstrip("s") or g + "es" == w or g == w + "es"
    return g == w


def score(item: Item, response: str) -> bool:
    """Anahtarlar birebir, degerler tip ve icerik olarak dogru olmali."""
    got, want = extract_json(response), json.loads(item.answer)
    if got is None or set(got) != set(want):
        return False
    for k, v in want.items():
        g = got[k]
        if isinstance(v, str):
            if not isinstance(g, str):
                return False
            ok = _same_product(g, v, item.lang) if k == "product" else g.strip() == v
            if not ok:
                return False
        elif isinstance(g, bool) or not isinstance(g, (int, float)) or abs(g - v) > 1e-9:
            return False
    return True
