"""Paylasilan sunucudaki uretim servisini koruyan bekci.

Iki parca:
  1) Saf mantik: check() ve preflight() -> sadece veri alir, ihlal listesi dondurur.
     GPU'suz test edilebilir (Windows, CI).
  2) Okuyucular: nvidia-smi, disk, servis HTTP -> sadece sunucuda calisir.

Kullanim (sunucuda):
  python -m trbench.guard --preflight
  python -m trbench.guard --stop-cmd "docker stop trbench-engine"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import httpx

MIB_PER_GB = 1024


@dataclass(frozen=True)
class GuardConfig:
    protected_gpus: tuple[int, ...] = (0, 1, 3)  # baska islere ait kartlar
    bench_gpu: int = 2
    # Korunan GPU'larda izin verilen oynama. GPU 0-1'deki is bellegini basta ayirir,
    # oynamaz. GPU 3'teki is dogal olarak dalgalanir (olculen: 25.0-26.5 GB).
    gpu_tolerance_mib: dict[int, int] = field(default_factory=lambda: {0: 1024, 1: 1024, 3: 4096})
    # GPU 2'deki servis bir modeli ilk istekte yukler (1.5 -> 3.7 GB, tepe 7 GB).
    # Ilk gece bos alan 767 MiB'a dustu; esik 2 GB.
    bench_gpu_min_free_mib: int = 2048
    # Servisin kendi izleme sistemi GPU 2 TOPLAMI > 20000 MiB'da alarm verir.
    # Biz her zaman ondan ONCE duruyoruz: 17000 MiB.
    bench_gpu_max_total_mib: int = 17000
    # Oturum oncesi: GPU 2'de servisin normal yukunden fazlasi olmamali.
    preflight_bench_max_used_mib: int = 5000
    min_root_free_gb: float = 100.0
    max_disk3_used_pct: float = 85.0
    # Korunan servisin saglik adresi: koda degil ortam degiskenine yazilir.
    service_url: str = field(default_factory=lambda: os.environ.get(
        "TRBENCH_SERVICE_HEALTH", "http://127.0.0.1:8000/health"))
    latency_factor: float = 3.0          # baseline'in kac kati alarm
    latency_floor_ms: float = 500.0      # kucuk baseline'larda yanlis alarm olmasin
    interval_s: int = 30
    strikes: int = 2                     # ust uste kac ihlalde durdur


@dataclass
class Snapshot:
    gpu_used_mib: dict[int, int]
    gpu_free_mib: dict[int, int]
    root_free_gb: float
    disk3_used_pct: float
    service_ok: bool
    service_latency_ms: float | None
    taken_at: float = field(default_factory=time.time)


# ---------- 1) Saf mantik ----------

def check(baseline: Snapshot, now: Snapshot, cfg: GuardConfig) -> list[str]:
    """Benchmark sirasinda: baseline'a gore ihlalleri dondurur. Bos liste = guvenli."""
    v: list[str] = []

    for g in cfg.protected_gpus:
        before, after = baseline.gpu_used_mib.get(g), now.gpu_used_mib.get(g)
        if after is None:
            v.append(f"GPU {g} okunamadi")
        elif before is not None and abs(after - before) > cfg.gpu_tolerance_mib.get(g, 1024):
            v.append(f"GPU {g} bellegi degisti: {before} -> {after} MiB")

    free2 = now.gpu_free_mib.get(cfg.bench_gpu)
    if free2 is not None and free2 < cfg.bench_gpu_min_free_mib:
        v.append(f"GPU {cfg.bench_gpu} bos bellek kritik: {free2} MiB")
    used2 = now.gpu_used_mib.get(cfg.bench_gpu)
    if used2 is not None and used2 > cfg.bench_gpu_max_total_mib:
        v.append(f"GPU {cfg.bench_gpu} toplam {used2} MiB > {cfg.bench_gpu_max_total_mib} "
                 f"(servis alarmi 20000'de)")

    if now.root_free_gb < cfg.min_root_free_gb:
        v.append(f"Root disk bos alan dusuk: {now.root_free_gb:.0f} GB")
    if now.disk3_used_pct > cfg.max_disk3_used_pct:
        v.append(f"/disk3 doluluk %{now.disk3_used_pct:.0f}")

    if not now.service_ok:
        v.append("Servis yanit vermiyor")
    elif baseline.service_latency_ms is not None and now.service_latency_ms is not None:
        limit = max(baseline.service_latency_ms * cfg.latency_factor, cfg.latency_floor_ms)
        if now.service_latency_ms > limit:
            v.append(f"Servis gecikmesi {now.service_latency_ms:.0f} ms (limit {limit:.0f})")

    return v


def preflight(now: Snapshot, cfg: GuardConfig) -> list[str]:
    """Oturum oncesi: biri tutmazsa o gun sunucu kullanilmaz."""
    v: list[str] = []
    used2 = now.gpu_used_mib.get(cfg.bench_gpu, 10**9)
    if used2 > cfg.preflight_bench_max_used_mib:
        v.append(f"GPU {cfg.bench_gpu} zaten {used2} MiB kullaniliyor "
                 f"(> {cfg.preflight_bench_max_used_mib}); servis yogun olabilir")
    if now.root_free_gb < cfg.min_root_free_gb:
        v.append(f"Root disk bos alan {now.root_free_gb:.0f} GB < {cfg.min_root_free_gb:.0f}")
    if now.disk3_used_pct > cfg.max_disk3_used_pct:
        v.append(f"/disk3 doluluk %{now.disk3_used_pct:.0f}")
    if not now.service_ok:
        v.append("Servis saglikli degil")
    return v


def parse_nvidia_smi(csv_text: str) -> tuple[dict[int, int], dict[int, int]]:
    """'index, memory.used, memory.free' satirlarini (noheader,nounits) ayristirir."""
    used: dict[int, int] = {}
    free: dict[int, int] = {}
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        idx = int(parts[0])
        used[idx], free[idx] = int(parts[1]), int(parts[2])
    return used, free


# ---------- 2) Okuyucular (sadece sunucuda) ----------

def read_gpus() -> tuple[dict[int, int], dict[int, int]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20,
    ).stdout
    return parse_nvidia_smi(out)


def probe_service(url: str, timeout_s: float = 5.0) -> tuple[bool, float | None]:
    start = time.perf_counter()
    try:
        ok = httpx.get(url, timeout=timeout_s).status_code == 200
    except httpx.HTTPError:
        return False, None
    return ok, (time.perf_counter() - start) * 1000


def take_snapshot(cfg: GuardConfig) -> Snapshot:
    used, free = read_gpus()
    root = shutil.disk_usage("/")
    d3 = shutil.disk_usage("/disk3")
    ok, lat = probe_service(cfg.service_url)
    return Snapshot(
        gpu_used_mib=used,
        gpu_free_mib=free,
        root_free_gb=root.free / 1024**3,
        disk3_used_pct=d3.used / d3.total * 100,
        service_ok=ok,
        service_latency_ms=lat,
    )


def watch(cfg: GuardConfig, stop_cmd: str) -> None:
    baseline = take_snapshot(cfg)
    print(f"[guard] baseline alindi, GPU kullanim: {baseline.gpu_used_mib}", flush=True)
    strikes = 0
    while True:
        time.sleep(cfg.interval_s)
        violations = check(baseline, take_snapshot(cfg), cfg)
        strikes = strikes + 1 if violations else 0
        if violations:
            print(f"[guard] ihlal ({strikes}/{cfg.strikes}): {violations}", flush=True)
        if strikes >= cfg.strikes:
            print(f"[guard] DURDURULUYOR -> {stop_cmd}", flush=True)
            subprocess.run(stop_cmd, shell=True, check=False)
            raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paylasilan sunucu bekcisi")
    ap.add_argument("--preflight", action="store_true", help="Oturum oncesi kontrol")
    ap.add_argument("--stop-cmd", help="Ihlalde calisacak komut (orn. docker stop ...)")
    args = ap.parse_args()
    cfg = GuardConfig()

    if args.preflight:
        problems = preflight(take_snapshot(cfg), cfg)
        for p in problems:
            print("HAYIR:", p)
        print("UYGUN: sunucu kullanilabilir" if not problems else "BUGUN SUNUCU KULLANILMAZ")
        raise SystemExit(1 if problems else 0)

    if not args.stop_cmd:
        ap.error("--stop-cmd gerekli (ya da --preflight)")
    watch(cfg, args.stop_cmd)


if __name__ == "__main__":
    main()
