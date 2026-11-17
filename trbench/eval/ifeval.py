"""Talimat takibi: IFEval tarzi, TR/EN paralel, sablondan uretilen sorular.

Neden kendi setimiz:
  - Paralellik garanti: ayni konu + ayni kisit, iki dilde.
  - Kontaminasyon yok: sorular bu repoda uretiliyor, hicbir egitim verisinde yok.
  - Lisans bizim.
Her kisit deterministik bir fonksiyonla kontrol edilir (LLM-judge yok).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from trbench.eval.schema import Item
from trbench.metrics.text import count_words

SEED = 2026

# (TR kok, EN kelime). TR'de 'cay' -> 'cayir', 'dag' -> 'dagitim' gibi on ek
# cakismasi olan kelimeler bilerek secilmedi (asagidaki keyword kontrolune bak).
TOPICS = [
    ("deniz", "sea"), ("kitap", "book"), ("şehir", "city"), ("müzik", "music"),
    ("kahve", "coffee"), ("bilgisayar", "computer"), ("orman", "forest"),
    ("nehir", "river"), ("okul", "school"), ("güneş", "sun"), ("bahçe", "garden"),
    ("köprü", "bridge"),
]
TASK = {"tr": "{t} hakkında bir metin yaz.", "en": "Write a text about the {t}."}
END_PHRASE = {"tr": "Başka sorunuz var mı?", "en": "Any other questions?"}

DESC = {
    "min_words": {"tr": "Cevabın en az {n} kelime olsun.",
                  "en": "Your response must contain at least {n} words."},
    "max_words": {"tr": "Cevabın en fazla {n} kelime olsun.",
                  "en": "Your response must contain at most {n} words."},
    "no_commas": {"tr": "Cevabında hiç virgül kullanma.",
                  "en": "Do not use any commas in your response."},
    "lowercase": {"tr": "Cevabının tamamı küçük harfle yazılsın, hiç büyük harf kullanma.",
                  "en": "Your entire response must be in lowercase letters, no capitals."},
    "keyword": {"tr": "'{kw}' kelimesini en az {n} kez kullan.",
                "en": "Use the word '{kw}' at least {n} times."},
    "end_with": {"tr": "Cevabını tam olarak şu ifadeyle bitir: {p}",
                 "en": "End your response with exactly this phrase: {p}"},
    "bullets": {"tr": "Cevabın tam olarak {n} madde işaretinden oluşsun. "
                      "Markdown kullan, örneğin: * Bu bir madde.",
                "en": "Your answer must contain exactly {n} bullet points. "
                      "Use markdown, for example: * This is a point."},
    "paragraphs": {"tr": "Cevabın {n} paragraftan oluşsun. Paragrafları *** ile ayır.",
                   "en": "There should be {n} paragraphs. Separate them with ***."},
    "title": {"tr": "Cevabına <<başlık>> biçiminde, çift açılı ayraç içinde bir başlık ekle.",
              "en": "Include a title wrapped in double angular brackets, such as <<title>>."},
}
SINGLE = list(DESC)
# Birbiriyle celismeyen ikililer (orn. min+max kelime veya madde+paragraf yok)
PAIRS = [("no_commas", "max_words"), ("lowercase", "title"), ("keyword", "min_words"),
         ("end_with", "no_commas"), ("bullets", "lowercase"), ("paragraphs", "keyword"),
         ("title", "end_with"), ("max_words", "keyword")]
PARAMS = {"min_words": {"n": 120}, "max_words": {"n": 60}, "keyword": {"n": 3},
          "bullets": {"n": 3}, "paragraphs": {"n": 3}}


# ---------- Turkce'ye duyarli yardimcilar ----------

def tr_lower(s: str) -> str:
    """Python'un .lower() fonksiyonu 'I'yi 'i' yapar; Turkcede 'ı' olmali."""
    return s.replace("I", "ı").replace("İ", "i").lower()


def capitalize(s: str, lang: str) -> str:
    """Ilk harfi buyut. TR: 'i' -> 'İ' (Python 'I' yapardi)."""
    if not s:
        return s
    first = {"i": "İ", "ı": "I"}.get(s[0], s[0].upper()) if lang == "tr" else s[0].upper()
    return first + s[1:]


def has_upper(s: str, lang: str) -> bool:
    low = tr_lower(s) if lang == "tr" else s.lower()
    return low != s


def keyword_count(text: str, kw: str, lang: str) -> int:
    """TR: kelime kokle baslar ve devami ek olabilir ('denize', 'deniz'in').
    EN: kelimenin kendisi veya cogulu ('book', 'books')."""
    words = re.findall(r"\w+(?:['’]\w+)*", text)
    if lang == "tr":
        k = tr_lower(kw)
        return sum(1 for w in words if tr_lower(w).startswith(k))
    k = kw.lower()
    return sum(1 for w in words if w.lower() in (k, k + "s", k + "es"))


# ---------- Kisit kontrolleri ----------

def check(kind: str, response: str, lang: str, kw: str = "", **p) -> bool:
    r = response.strip()
    if kind == "min_words":
        return count_words(r) >= p["n"]
    if kind == "max_words":
        return count_words(r) <= p["n"]
    if kind == "no_commas":
        return "," not in r
    if kind == "lowercase":
        return not has_upper(r, lang)
    if kind == "keyword":
        return keyword_count(r, kw, lang) >= p["n"]
    if kind == "end_with":
        return r.endswith(END_PHRASE[lang])
    if kind == "bullets":
        return len(re.findall(r"^\s*[*-]\s", r, flags=re.MULTILINE)) == p["n"]
    if kind == "paragraphs":
        parts = [x for x in re.split(r"\n\s*\*\*\*\s*\n", r) if x.strip()]
        return len(parts) == p["n"]
    if kind == "title":
        return re.search(r"<<[^\n]+>>", r) is not None
    raise ValueError(f"bilinmeyen kisit: {kind}")


@dataclass(frozen=True)
class Spec:
    kinds: tuple[str, ...]
    topic: int


def specs(n: int = 60) -> list[Spec]:
    """Ilk yari tek kisit (her tur dengeli), ikinci yari uyumlu ikililer."""
    rng = random.Random(SEED)
    half = n // 2
    singles = [(SINGLE[i % len(SINGLE)],) for i in range(half)]
    pairs = [PAIRS[i % len(PAIRS)] for i in range(n - half)]
    return [Spec(k, rng.randrange(len(TOPICS))) for k in singles + pairs]


def ifeval_items(n: int = 60) -> list[Item]:
    items = []
    for idx, sp in enumerate(specs(n)):
        for li, lang in enumerate(("tr", "en")):
            topic = TOPICS[sp.topic][li]
            lines = [capitalize(TASK[lang].format(t=topic), lang)]
            for k in sp.kinds:
                lines.append(DESC[k][lang].format(kw=topic, p=END_PHRASE[lang],
                                                  **PARAMS.get(k, {})))
            items.append(Item(
                id=f"instruction/{idx:03d}",
                category="instruction",
                lang=lang,
                prompt=" ".join(lines),
                answer="",
                scorer="ifeval",
                source="trbench (ozgun sablon)",
                meta={"constraints": [{"kind": k, "kw": topic, **PARAMS.get(k, {})}
                                      for k in sp.kinds]},
            ))
    return items


def score(item: Item, response: str) -> bool:
    """Prompt duzeyinde katı puan: TUM kisitlar saglanmali."""
    return all(check(c["kind"], response, item.lang,
                     **{k: v for k, v in c.items() if k != "kind"})
               for c in item.meta["constraints"])
