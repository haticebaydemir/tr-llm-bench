"""Eval setinin ortak formati: her kaynak bu yapiya cevrilir."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

CATEGORIES = ("instruction", "json", "morphology", "reading", "knowledge", "math")
LANGS = ("tr", "en")


@dataclass(frozen=True)
class Item:
    id: str            # TR ve EN'de AYNI -> paralel eslesme bu alanla
    category: str
    lang: str
    prompt: str
    answer: str        # dogru cevap (harf, sayi, referans...)
    scorer: str        # hangi puanlayici: "choice", "number", "ifeval", "json", "morph"
    source: str        # nereden geldi (lisans/atif icin)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"bilinmeyen kategori: {self.category}")
        if self.lang not in LANGS:
            raise ValueError(f"bilinmeyen dil: {self.lang}")

    def to_dict(self) -> dict:
        return asdict(self)
