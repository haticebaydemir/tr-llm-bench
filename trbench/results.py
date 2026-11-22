"""Sonuc dosyalarindan rapora girecek satirlari secer.

Ayni grup (orn. bir model etiketi) birden fazla deney kimligiyle (exp) olculduyse
dosyadaki EN SON kimlik kullanilir; ayni soru/kosu iki kez varsa sonuncusu tutulur.
Ikisinde de stderr'e uyari yazilir, tablolar (stdout) degismez.
Eski satirlarda exp yok (None): tek kimlik gibi davranir, eski tablolar aynen cikar.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence


def _warn(msg: str) -> None:
    print(f"[uyari] {msg}", file=sys.stderr)


def pick(rows: list[dict], group: Sequence[str], item: Sequence[str],
         warn: Callable[[str], None] = _warn) -> list[dict]:
    latest: dict[tuple, object] = {}
    exps: dict[tuple, set] = {}
    for r in rows:
        g = tuple(r.get(k) for k in group)
        latest[g] = r.get("exp")
        exps.setdefault(g, set()).add(r.get("exp"))
    for g, e in exps.items():
        if len(e) > 1:
            warn(f"{'/'.join(map(str, g))}: {len(e)} deney kimligi var, sonuncusu kullaniliyor ({latest[g]})")
    keep: dict[tuple, int] = {}
    for i, r in enumerate(rows):
        g = tuple(r.get(k) for k in group)
        if r.get("exp") == latest[g]:
            keep[g + tuple(r.get(k) for k in item)] = i
    n_kept = sum(1 for r in rows if r.get("exp") == latest[tuple(r.get(k) for k in group)])
    if n_kept > len(keep):
        warn(f"{n_kept - len(keep)} tekrarlanan satir: her biri icin sonuncusu kullaniliyor")
    return [rows[i] for i in sorted(keep.values())]
