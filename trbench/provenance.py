"""Her sonuc satirinin 'hangi kod, hangi veri, hangi agirlik' bilgisi.

Amac: bir sonuc satirindan, onu ureten ortami birebir geri kurabilmek.
  git_commit  : kodun surumu (+ kaydedilmemis degisiklik varsa dirty=True)
  *_sha256    : veri seti ve agirlik dosyasinin icerik ozeti (ayni isim, farkli dosya olmasin)
  exp         : deney kimligi; sonucu degistirebilecek her seyin ozeti (resume ve raporlar bunu kullanir)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str | None:
    """Buyuk dosyalar (GGUF) icin sonuc yanina '.sha256' onbellegi; dosya degisirse yeniden hesaplanir."""
    path = Path(path)
    if not path.is_file():
        return None
    st = path.stat()
    # boyut + nanosaniye mtime + inode: ayni saniyede ayni boyutla degisen dosya da yakalanir
    stamp = f"{st.st_size}:{st.st_mtime_ns}:{st.st_ino}"
    cache = path.with_name(path.name + ".sha256")
    try:
        old_stamp, old = cache.read_text().split()
        if old_stamp == stamp:
            return old
    except (OSError, ValueError):
        pass
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()
    try:
        cache.write_text(f"{stamp} {digest}\n")
    except OSError:
        pass
    return digest


def git_commit() -> tuple[str, bool]:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(ROOT), *a], check=False,
                              capture_output=True, text=True).stdout.strip()
    sha = git("rev-parse", "HEAD")
    return (sha or "unknown"), bool(git("status", "--porcelain", "--untracked-files=no"))


def provenance() -> dict:
    sha, dirty = git_commit()
    return {"git_commit": sha, "git_dirty": dirty, "python": sys.version.split()[0],
            "platform": platform.platform()}


def experiment_id(parts: dict) -> str:
    """Deney kimligi: verilen parcalarin (ayar ozeti, eval seti, agirlik, engine surumu, kod surumu)
    kararli ozeti. Herhangi biri degisirse eski sonuclar 'bitmis' sayilmaz."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]
