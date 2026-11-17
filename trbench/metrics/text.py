import re

# \w Python 3'te Unicode'dur: c, g, i, s, o, u, I harflerini tanir.
# (?:['\u2019]\w+)* -> kesme isaretli ekleri ayni kelimeye baglar: "Ankara'ya" = 1 kelime.
# \d+(?:[.,]\d+)* -> "1.250" ve "3,5" tek kelime sayilir.
_WORD = re.compile(r"\d+(?:[.,]\d+)*|\w+(?:['\u2019]\w+)*", re.UNICODE)


def count_words(text: str) -> int:
    """Turkce ve Ingilizce metinde kelime sayisi."""
    return len(_WORD.findall(text))
