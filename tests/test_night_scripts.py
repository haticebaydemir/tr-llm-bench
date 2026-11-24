"""Faz 7r: gece script'leri bir adim hata verince rapor uretmez, HATA ile ve kod 1 ile biter."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NIGHT = sorted(ROOT.glob("scripts/night_*.sh"))


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                    reason="bash script davranisi Linux CI'da test edilir")
@pytest.mark.parametrize("script", ["night_llamacpp_cold.sh", "night_mnbt.sh", "night_6d.sh"])
def test_failed_step_ends_with_error(tmp_path, script):
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / script, tmp_path / "scripts" / script)
    (tmp_path / "scripts" / "gpu_bench.sh").write_text("exit 1\n")          # her olcum adimi basarisiz
    r = subprocess.run(["bash", str(tmp_path / "scripts" / script)], capture_output=True, text=True, check=False)
    last = r.stdout.strip().splitlines()[-1]
    assert r.returncode == 1 and "HATA" in last and "HEPSI BITTI" not in r.stdout


def test_every_night_script_has_failure_flag():
    assert NIGHT
    for p in NIGHT + [ROOT / "scripts/queue_models.sh"]:
        s = p.read_text(encoding="utf-8")
        assert "FAIL=0" in s and "exit 1" in s, p.name


# Faz 7r: rapor komutu basarisiz olursa da HATA + kod 1. Sahte python N. cagrida kod 1 doner (0 = hic).
REPORT_CALLS = {"night_llamacpp_cold.sh": 1, "night_llamacpp_cache.sh": 1, "night_mnbt.sh": 2}


def _run_with_fake_python(tmp_path, script, fail_on_call):
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / script, tmp_path / "scripts" / script)
    (tmp_path / "scripts" / "gpu_bench.sh").write_text("exit 0\n")          # olcum adimlari basarili
    calls = tmp_path / "calls"
    fake = tmp_path / "fake_python"
    fake.write_text(f'#!/usr/bin/env bash\necho x >> "{calls}"\n'
                    f'[ "$(wc -l < "{calls}")" -eq {fail_on_call} ] && exit 1\nexit 0\n')
    fake.chmod(0o755)
    env = {**os.environ, "TRBENCH_PY": str(fake)}
    r = subprocess.run(["bash", str(tmp_path / "scripts" / script)], capture_output=True, text=True,
                       check=False, env=env)
    n = len(calls.read_text().splitlines()) if calls.exists() else 0
    return r, n


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                    reason="bash script davranisi Linux CI'da test edilir")
@pytest.mark.parametrize("script,fail_on_call",
                         [(s, i) for s, n in REPORT_CALLS.items() for i in range(1, n + 1)])
def test_failed_report_ends_with_error(tmp_path, script, fail_on_call):
    r, n = _run_with_fake_python(tmp_path, script, fail_on_call)
    last = r.stdout.strip().splitlines()[-1]
    assert r.returncode == 1 and "HATA" in last and "HEPSI BITTI" not in r.stdout
    assert n == REPORT_CALLS[script]                                         # diger raporlar yine denenir


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                    reason="bash script davranisi Linux CI'da test edilir")
@pytest.mark.parametrize("script", sorted(REPORT_CALLS))
def test_successful_reports_end_with_done(tmp_path, script):
    r, n = _run_with_fake_python(tmp_path, script, 0)
    assert r.returncode == 0 and "HEPSI BITTI" in r.stdout.strip().splitlines()[-1] and "HATA" not in r.stdout
    assert n == REPORT_CALLS[script]
