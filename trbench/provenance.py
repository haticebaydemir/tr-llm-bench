"""Her sonuc satirinin 'hangi kod, hangi veri, hangi agirlik' bilgisi.

Amac: bir sonuc satirindan, onu ureten ortami birebir geri kurabilmek.
  git_commit  : kodun surumu (+ results/ disinda kaydedilmemis degisiklik varsa dirty=True)
  code_diff   : trbench/configs/scripts'te kaydedilmemis degisikligin ozeti; varsa GPU kosusu baslamaz
                (--allow-dirty ile acilirsa bu ozet deney kimligine girer)
  *_sha256    : veri seti ve agirlik dosyasinin icerik ozeti (ayni isim, farkli dosya olmasin)
  exp         : deney kimligi; sonucu degistirebilecek her seyin ozeti (resume ve raporlar bunu kullanir)
"""

from __future__ import annotations

import hashlib
import json
import os
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


CODE_PATHS = ("trbench", "configs", "scripts")


def _git(root: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(root), *a], check=False,
                          capture_output=True, text=True).stdout.strip()


def git_commit(root: Path = ROOT) -> tuple[str, bool]:
    """Kosu sonuclari takip edilen results/*.jsonl'e yaziliyor: o dosyalar 'kirli' sayilmaz,
    yoksa her kosu bir sonrakini kirli gosterir."""
    sha = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results")
    return (sha or "unknown"), bool(status)


def code_diff_sha(root: Path = ROOT) -> str | None:
    """Kod, config ve script'teki kaydedilmemis degisikligin ozeti; temizse None."""
    d = _git(root, "diff", "HEAD", "--", *CODE_PATHS)
    return sha256_text(d) if d else None


def require_clean(allow_dirty: bool, root: Path = ROOT) -> str | None:
    """Kodda kaydedilmemis degisiklik varsa kosu baslamaz. --allow-dirty ile baslarsa
    degisikligin ozeti dondurulur ve deney kimligine girer."""
    d = code_diff_sha(root)
    if d and not allow_dirty:
        raise SystemExit("[provenance] trbench/configs/scripts'te commit edilmemis degisiklik var: "
                         "once commit et (ya da bilerek --allow-dirty)")
    return d


def hardware() -> dict:
    """CPU modeli, RAM ve kernel: CPU olcumlerinin (llama-bench, CPU kalite) yorumu icin."""
    cpu = platform.processor() or ""
    ram_gb = None
    try:
        txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        cpu = next((ln.split(":", 1)[1].strip() for ln in txt.splitlines() if ln.startswith("model name")), cpu)
        mem = Path("/proc/meminfo").read_text(encoding="utf-8").split()
        ram_gb = round(int(mem[mem.index("MemTotal:") + 1]) / 1024**2, 1)
    except (OSError, ValueError, StopIteration):
        pass
    return {"cpu": cpu or None, "cpu_count": os.cpu_count(), "ram_gb": ram_gb, "kernel": platform.release()}


def provenance() -> dict:
    sha, dirty = git_commit()
    return {"git_commit": sha, "git_dirty": dirty, "python": sys.version.split()[0],
            "platform": platform.platform(), **hardware()}


def experiment_id(parts: dict) -> str:
    """Deney kimligi: verilen parcalarin (ayar ozeti, eval seti, agirlik, engine surumu, kod surumu)
    kararli ozeti. Herhangi biri degisirse eski sonuclar 'bitmis' sayilmaz."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]
