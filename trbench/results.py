"""Sonuc dosyalarindan rapora girecek satirlari secer.

Ayni grup (orn. bir model etiketi) birden fazla deney kimligiyle (exp) olculduyse
EN YENI kimlik kullanilir; ayni soru/kosu iki kez varsa en yenisi tutulur. "En yeni" satirdaki
ts (ISO zaman) alanina gore belirlenir, dosya sirasina gore degil: birlestirilmis ya da sirasi
bozulmus dosyada da dogru kalir. ts olmayan eski satirlarda dosya sirasi kullanilir.
Ikisinde de stderr'e uyari yazilir, tablolar (stdout) degismez.
Eski satirlarda exp yok (None): tek kimlik gibi davranir, eski tablolar aynen cikar.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence


def _warn(msg: str) -> None:
    print(f"[uyari] {msg}", file=sys.stderr)


def _order(i: int, r: dict) -> tuple:
    """Siralama anahtari: once ts (yoksa en eski sayilir), esitlikte dosya sirasi."""
    return (r.get("ts") or "", i)


def pick(rows: list[dict], group: Sequence[str], item: Sequence[str],
         warn: Callable[[str], None] = _warn) -> list[dict]:
    latest: dict[tuple, object] = {}
    newest: dict[tuple, tuple] = {}
    exps: dict[tuple, set] = {}
    for i, r in enumerate(rows):
        g = tuple(r.get(k) for k in group)
        o = _order(i, r)
        if g not in newest or o >= newest[g]:
            newest[g], latest[g] = o, r.get("exp")
        exps.setdefault(g, set()).add(r.get("exp"))
    for g, e in exps.items():
        if len(e) > 1:
            warn(f"{'/'.join(map(str, g))}: {len(e)} deney kimligi var, sonuncusu kullaniliyor ({latest[g]})")
    keep: dict[tuple, int] = {}
    for i, r in enumerate(rows):
        g = tuple(r.get(k) for k in group)
        if r.get("exp") == latest[g]:
            key = g + tuple(r.get(k) for k in item)
            if key not in keep or _order(i, r) >= _order(keep[key], rows[keep[key]]):
                keep[key] = i
    n_kept = sum(1 for r in rows if r.get("exp") == latest[tuple(r.get(k) for k in group)])
    if n_kept > len(keep):
        warn(f"{n_kept - len(keep)} tekrarlanan satir: her biri icin sonuncusu kullaniliyor")
    return [rows[i] for i in sorted(keep.values())]
