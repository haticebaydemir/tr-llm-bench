"""Tum puanlayicilar tek kapidan: score(item, response) -> bool.

choice/number burada; ifeval/json/morph kendi modullerinde (o kategorinin bilgisi orada).
"""

from __future__ import annotations

import re

from trbench.eval import ifeval, json_tasks, morphology
from trbench.eval.schema import Item

_LABEL = r"(?:cevap|yanıt|answer)\s*[:：]?\s*"


def parse_choice(response: str) -> str | None:
    """'Cevap: B', 'B)', '**B**', 'B şıkkı' -> 'B'.

    Oncelik: 1) 'Cevap/Answer: X' etiketi  2) cevap tek harfle basliyorsa
             3) parantez/nokta ile biten ilk harf. Ingilizce 'A' artikelini
             yakalamamak icin ciplak 'A' kelimesi tek basina sayilmaz.
    """
    r = response.strip().replace("*", "")
    m = re.search(_LABEL + r"\(?([ABCD])\b", r, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r"\(?([ABCD])(?:[).:]|\s+(?:şık|seçenek|option)|$)", r, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABCD])\)", r)
    return m.group(1) if m else None


_NUM = r"-?\d[\d.,]*"


def to_number(raw: str, lang: str) -> float | None:
    """Dile gore sayi okuma. TR: 1.250,75 | EN: 1,250.75"""
    s = raw.strip().rstrip(".,")
    if lang == "tr":
        if re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", s):
            s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_number(response: str, lang: str) -> float | None:
    """Once 'Cevap/Answer: N' etiketine, yoksa metindeki SON sayiya bakar."""
    m = re.findall(_LABEL + r"\$?\s*(" + _NUM + ")", response, flags=re.IGNORECASE)
    raw = m[-1] if m else (re.findall(_NUM, response) or [None])[-1]
    return to_number(raw, lang) if raw else None


def score(item: Item, response: str) -> bool:
    if item.scorer == "choice":
        return parse_choice(response) == item.answer
    if item.scorer == "number":
        got = parse_number(response, item.lang)
        return got is not None and abs(got - float(item.answer)) < 1e-6
    if item.scorer == "ifeval":
        return ifeval.score(item, response)
    if item.scorer == "json":
        return json_tasks.score(item, response)
    if item.scorer == "morph":
        return morphology.score(item, response)
    raise ValueError(f"bilinmeyen puanlayici: {item.scorer}")
