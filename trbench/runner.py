"""Faz 3 calistiricisi: config'teki her model icin vLLM'i GPU 2'de baslatir,
tum (dil x esanlilik x tekrar) kosularini yapar, sonucu satir satir kaydeder.

Gece boyunca gozetimsiz calisacagi icin:
  - Her sonuc aninda diske yazilir (jsonl, append) -> cokse bile kayip yok.
  - Yeniden baslatilinca bitmis kosulari ATLAR (resume).
  - Bir model inmezse / baslamazsa kaydedilir ve SIRADAKI modele gecilir.
  - Guard vLLM'i oldururse runner bunu fark eder ve durur.

Kullanim (sunucuda, scripts/phase3.sh uzerinden):
  python -m trbench.runner configs/phase3.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from trbench.config import ModelSpec, Phase3, load
from trbench.load.generator import run as load_run
from trbench.load.generator import summarize

ROOT = Path("/disk3/trbench")
MODELS_DIR = ROOT / "models"
LOGS = ROOT / "logs"
VLLM = ROOT / "venv-vllm/bin/vllm"
RESULTS = Path("results/phase3.jsonl")
EVAL = Path("data/eval/eval_set.jsonl")
KEY_FIELDS = ("model", "quant", "lang", "concurrency", "repeat")


# ---------- Saf yardimcilar (testli) ----------

def run_key(row: dict) -> tuple:
    return tuple(row[k] for k in KEY_FIELDS)


def done_keys(path: Path) -> set[tuple]:
    """Daha once BASARIYLA bitmis kosular (resume icin)."""
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") == "ok":
            keys.add(run_key(row))
    return keys


def prompts_for(items: list[dict], lang: str, n: int) -> list[str]:
    """O dilin prompt'larindan n tane; yetmezse basa sarar (deterministik)."""
    pool = [i["prompt"] for i in items if i["lang"] == lang]
    return list(itertools.islice(itertools.cycle(pool), n))


def served_name(m: ModelSpec) -> str:
    return f"trbench-{m.name}-{m.quant}"   # guard'in pkill desenine uyar


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- Sunucu tarafi (GPU) ----------

def env_meta() -> dict:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                          "--format=csv,noheader", "-i", "2"],
                         check=False, capture_output=True, text=True).stdout.strip()
    ver = subprocess.run([str(VLLM.parent / "python"), "-c", "import vllm;print(vllm.__version__)"],
                         check=False, capture_output=True, text=True).stdout.strip()
    return {"gpu": smi, "vllm": ver, "host": platform.node()}


def download(m: ModelSpec) -> tuple[Path, str]:
    from huggingface_hub import HfApi, snapshot_download

    rev = HfApi().model_info(m.repo).sha
    path = snapshot_download(m.repo, revision=rev,
                             local_dir=MODELS_DIR / f"{m.name}-{m.quant}",
                             allow_patterns=["*.json", "*.safetensors", "*.model",
                                             "*.txt", "*.jinja", "*.py"])
    return Path(path), rev


def start_vllm(cfg: Phase3, m: ModelSpec, path: Path) -> subprocess.Popen:
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="2")
    cmd = [str(VLLM), "serve", str(path), "--served-model-name", served_name(m),
           "--host", "127.0.0.1", "--port", str(cfg.port), "--dtype", "half",
           "--max-model-len", str(cfg.max_model_len),
           "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
           "--max-num-seqs", str(cfg.max_num_seqs)]
    log = (LOGS / f"vllm_{served_name(m)}.log").open("w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_healthy(proc: subprocess.Popen, port: int, timeout_s: int = 600) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:        # surec oldu (desteklenmeyen quant vb.)
            return False
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(5)
    return False


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/phase3.yaml")
    cfg = load(ap.parse_args().config)
    items = [json.loads(x) for x in EVAL.read_text(encoding="utf-8").splitlines()]
    done = done_keys(RESULTS)
    meta = env_meta()
    LOGS.mkdir(parents=True, exist_ok=True)

    for m in cfg.models:
        todo = [r for r in cfg.runs() if (r["model"], r["quant"]) == (m.name, m.quant)
                and run_key(r) not in done]
        if not todo:
            print(f"[runner] {m.name}/{m.quant}: hepsi bitmis, atla", flush=True)
            continue
        base = {"model": m.name, "quant": m.quant, "repo": m.repo, **meta}
        try:
            path, rev = download(m)
        except Exception as e:  # noqa: BLE001 -- gated, silinmis repo: kaydet, devam
            append(RESULTS, {**base, "status": "download_failed", "error": str(e)[:300],
                             "lang": None, "concurrency": None, "repeat": None})
            print(f"[runner] {m.name}/{m.quant}: INDIRILEMEDI -> {e}", flush=True)
            continue

        proc = start_vllm(cfg, m, path)
        if not wait_healthy(proc, cfg.port):
            stop(proc)
            append(RESULTS, {**base, "revision": rev, "status": "engine_failed",
                             "lang": None, "concurrency": None, "repeat": None,
                             "error": f"log: vllm_{served_name(m)}.log"})
            print(f"[runner] {m.name}/{m.quant}: vLLM BASLAMADI (V100 destegi?)", flush=True)
            continue

        try:
            for r in todo:
                if proc.poll() is not None:
                    raise SystemExit("[runner] vLLM durdu (guard?) -> runner da duruyor")
                prompts = prompts_for(items, r["lang"], r["n_requests"])
                results, wall = asyncio.run(load_run(
                    prompts, f"http://127.0.0.1:{cfg.port}", served_name(m),
                    r["concurrency"], cfg.max_tokens))
                s = summarize(results, wall)
                append(RESULTS, {**base, **r, "revision": rev, "status": "ok", **s,
                                 "ts": datetime.now(timezone.utc).isoformat()})
                print(f"[runner] {m.name}/{m.quant} {r['lang']} c={r['concurrency']} "
                      f"#{r['repeat']}: {s['tokens_per_s']:.0f} tok/s, "
                      f"TTFT {s['ttft_p50_ms']:.0f} ms, hata {s['errors']}", flush=True)
        finally:
            stop(proc)


if __name__ == "__main__":
    main()
