"""GPU calistiricisi: config'teki her model icin engine'i (vLLM ya da llama.cpp) GPU 2'de baslatir,
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
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from trbench.config import ModelSpec, Phase3, load
from trbench.load.generator import run as load_run
from trbench.load.generator import summarize
from trbench.provenance import experiment_id, file_sha256, provenance, require_clean, sha256_text

ROOT = Path("/disk3/trbench")
MODELS_DIR = ROOT / "models"
LOGS = ROOT / "logs"
# TRBENCH_VLLM: baska bir vLLM kurulumu (ör. upstream) icin
VLLM = Path(os.environ.get("TRBENCH_VLLM", str(ROOT / "venv-vllm/bin/vllm")))
LLAMA_CUDA = ROOT / "llama.cpp/build-cuda/bin/llama-server"
EVAL = Path("data/eval/eval_set.jsonl")
# exp: tam deney kimligi (ayar ozeti fp + eval seti + agirlik + engine surumu + kod surumu)
KEY_FIELDS = ("engine", "model", "quant", "lang", "concurrency", "repeat", "fp", "exp")


# ---------- Saf yardimcilar (testli) ----------

def run_key(row: dict) -> tuple:
    return tuple(row.get(k, "vllm" if k == "engine" else None) for k in KEY_FIELDS)


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


def token_windows(tokens: list[int], size: int, n: int) -> list[list[int]]:
    """Uzun bir token dizisinden ardisik, ortusmeyen `size` uzunlugunda n pencere (yetmezse basa sarar)."""
    if len(tokens) < size:
        raise ValueError(f"{size} token'lik pencere icin metin cok kisa ({len(tokens)})")
    k = len(tokens) // size
    return [tokens[(i % k) * size:(i % k + 1) * size] for i in range(n)]


def matched_prompts(url: str, model: str, items: list[dict], lang: str, size: int, n: int) -> list[str]:
    """TR ve EN icin AYNI token sayisinda prompt: o dilin prompt'lari birlestirilir, engine'in kendi
    tokenizer'iyla (/tokenize) size'lik pencerelere bolunur, /detokenize ile metne doner.
    vLLM ve llama.cpp ayni endpoint adlarini farkli alanlarla kullanir; ikisi de desteklenir."""
    text = "\n\n".join(i["prompt"] for i in items if i["lang"] == lang)
    r = httpx.post(f"{url}/tokenize", json={"model": model, "prompt": text, "content": text,
                                            "add_special_tokens": False}, timeout=60).json()
    out = []
    for w in token_windows(r["tokens"], size, n):
        d = httpx.post(f"{url}/detokenize", json={"model": model, "tokens": w}, timeout=60).json()
        out.append(d.get("prompt", d.get("content", "")))
    return out


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
    lcpp = subprocess.run([str(LLAMA_CUDA), "--version"], check=False,
                          capture_output=True, text=True)
    lcpp_ver = next((x for x in (lcpp.stdout + lcpp.stderr).splitlines() if "version" in x), "")
    full = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True).stdout
    cuda = re.search(r"CUDA Version:\s*(\S+)", full)
    return {"gpu": smi, "cuda_driver": cuda.group(1) if cuda else None,
            "vllm": ver, "llamacpp": lcpp_ver.strip(), "host": platform.node(),
            **provenance()}


def gpu2_used_mib() -> int | None:
    out = subprocess.run(["nvidia-smi", "-i", "2", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         check=False, capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None


class PeakSampler:
    """Kosu boyunca GPU 2 TOPLAM bellegini (servis dahil) 1 sn arayla izler; tepeyi tutar."""

    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = gpu2_used_mib()
            if v:
                self.peak = max(self.peak, v)
            self._stop.wait(1.0)

    def __enter__(self) -> PeakSampler:  # noqa: PYI034
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join(timeout=5)


def download(m: ModelSpec, engine: str) -> tuple[Path, str]:
    if engine == "llamacpp":
        from trbench.eval.gguf import download_model
        return download_model(m.repo, m.quant), ("local" if Path(m.repo).is_dir() else m.repo)
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
    cmd += cfg.extra_args + os.environ.get("VLLM_EXTRA", "").split()
    if cfg.enforce_eager:
        cmd.append("--enforce-eager")
    log = (LOGS / f"vllm_{served_name(m)}.log").open("w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def llamacpp_cmd(cfg: Phase3, m: ModelSpec, gguf: Path) -> list[str]:
    nseq = max(cfg.conc(m))
    cmd = [str(LLAMA_CUDA), "-m", str(gguf), "--alias", served_name(m),
           "--host", "127.0.0.1", "--port", str(cfg.port), "-ngl", "99",
           "-c", str(cfg.max_model_len * nseq), "-np", str(nseq), "-t", "8"]
    if m.chat_template:
        cmd += ["--chat-template-file", m.chat_template]
    return cmd + cfg.extra_args


def start_llamacpp(cfg: Phase3, m: ModelSpec, gguf: Path) -> subprocess.Popen:
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="2")
    log = (LOGS / f"llamacpp_{served_name(m)}.log").open("w")
    return subprocess.Popen(llamacpp_cmd(cfg, m, gguf), env=env, stdout=log, stderr=subprocess.STDOUT)


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
    ap.add_argument("config", nargs="?", default="configs/gpu_llamacpp.yaml")
    ap.add_argument("--allow-dirty", action="store_true", help="commit edilmemis kodla kos (ozeti kimlige girer)")
    code_diff = require_clean(ap.parse_args().allow_dirty)
    cfg = load(ap.parse_args().config)
    out = Path(cfg.results)
    items = [json.loads(x) for x in EVAL.read_text(encoding="utf-8").splitlines()]
    done = done_keys(out)
    meta = {**env_meta(), "config": ap.parse_args().config,
            "config_sha": sha256_text(Path(ap.parse_args().config).read_text(encoding="utf-8")),
            "eval_sha": file_sha256(EVAL)}
    LOGS.mkdir(parents=True, exist_ok=True)

    for m in cfg.models:
        base = {"engine": cfg.engine, "model": m.name, "quant": m.quant, "repo": m.repo,
                "enforce_eager": cfg.enforce_eager, "gpu_memory_utilization": cfg.gpu_memory_utilization,
                "prompt_tokens_target": cfg.prompt_tokens, "extra_args": cfg.extra_args,
                "queue_metrics": cfg.queue_metrics, "cache_prompt": cfg.cache_prompt, **meta}
        # Kimlik agirliga bagli: kontrol model indirildikten (ya da yerelde bulunduktan) SONRA.
        try:
            path, rev = download(m, cfg.engine)
        except Exception as e:  # noqa: BLE001 -- gated, silinmis repo: kaydet, devam
            append(out, {**base, "status": "download_failed", "error": str(e)[:300],
                             "lang": None, "concurrency": None, "repeat": None})
            print(f"[runner] {m.name}/{m.quant}: INDIRILEMEDI -> {e}", flush=True)
            continue

        if cfg.engine == "llamacpp":
            base["weights_file"], base["weights_sha256"] = path.name, file_sha256(path)
        base["exp"] = experiment_id({
            "fp": cfg.fingerprint(m), "eval_sha": meta["eval_sha"], "revision": rev,
            "weights": base.get("weights_sha256"), "git_commit": meta["git_commit"],
            "engine_version": meta["vllm"] if cfg.engine == "vllm" else meta["llamacpp"],
            **({"code_diff": code_diff} if code_diff else {})})       # temizken eski kimlikler aynen
        todo = [r for r in cfg.runs() if (r["model"], r["quant"]) == (m.name, m.quant)
                and run_key({**r, "exp": base["exp"]}) not in done]
        if not todo:
            print(f"[runner] {m.name}/{m.quant}: hepsi bitmis (exp {base['exp']}), atla", flush=True)
            continue
        idle = gpu2_used_mib()
        proc = (start_llamacpp if cfg.engine == "llamacpp" else start_vllm)(cfg, m, path)
        if not wait_healthy(proc, cfg.port):
            stop(proc)
            append(out, {**base, "revision": rev, "status": "engine_failed",
                             "lang": None, "concurrency": None, "repeat": None,
                             "error": f"log: {cfg.engine}_{served_name(m)}.log"})
            print(f"[runner] {m.name}/{m.quant}: {cfg.engine} BASLAMADI", flush=True)
            continue

        url = f"http://127.0.0.1:{cfg.port}"
        try:
            if cfg.warmup:                       # ilk isteklerin derleme/onbellek etkisi olcume girmesin
                for lang in cfg.langs:
                    asyncio.run(load_run(prompts_for(items, lang, cfg.warmup), url, served_name(m), 1, 32))
            for r in todo:
                if proc.poll() is not None:
                    raise SystemExit("[runner] engine durdu (guard?) -> runner da duruyor")
                prompts = (matched_prompts(url, served_name(m), items, r["lang"], cfg.prompt_tokens,
                                           r["n_requests"]) if cfg.prompt_tokens
                           else prompts_for(items, r["lang"], r["n_requests"]))
                queue = {} if cfg.queue_metrics else None
                with PeakSampler() as ps:
                    results, wall = asyncio.run(load_run(
                        prompts, url, served_name(m), r["concurrency"], cfg.max_tokens, queue=queue,
                        extra=None if cfg.cache_prompt is None else {"cache_prompt": cfg.cache_prompt}))
                s = {**summarize(results, wall), **(queue or {})}
                append(out, {**base, **r, "revision": rev, "status": "ok", **s,
                                 "gpu2_idle_mib": idle, "gpu2_peak_mib": ps.peak,
                                 "ts": datetime.now(timezone.utc).isoformat()})
                print(f"[runner] {cfg.engine} {m.name}/{m.quant} {r['lang']} c={r['concurrency']} "
                      f"#{r['repeat']}: {s['tokens_per_s']:.0f} tok/s, TTFT {s['ttft_p50_ms']:.0f} ms, "
                      f"GPU2 tepe {ps.peak} MiB, hata {s['errors']}"
                      + (f", kuyruk max {s.get('queue_waiting_max')}, preempt {s.get('preemptions')}"
                         if queue is not None else ""), flush=True)
        finally:
            stop(proc)


if __name__ == "__main__":
    main()
