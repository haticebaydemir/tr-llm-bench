"""Faz 7r: gece script'leri bir adim hata verince rapor uretmez, HATA ile ve kod 1 ile biter."""
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
