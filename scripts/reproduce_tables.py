"""Faz 8g: README tablolarini repodaki sonuc dosyalarindan yeniden uret (GPU/model gerekmez).

  python scripts/reproduce_tables.py            # ciktilar: repro/<modul>.md
Temiz bir ortamda:  python -m venv .venv ; pip install -r env/requirements-server.lock ;
                    pip install -e . --no-deps ; python scripts/reproduce_tables.py
Her modul ayri calisir; biri hata verirse digerleri devam eder, sonda ozet ve cikis kodu 1.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODULES = [
    "trbench.tokenizer_tax.langs",
    "trbench.tokenizer_tax.distribution",
    "trbench.eval.analysis --all --clean",
    "trbench.eval.gpu_quality_report",
    "trbench.eval.kv_report",
    "trbench.eval.breakdown",
    "trbench.eval.stats",
    "trbench.eval.gpu_report",
    "trbench.eval.context_tax",
    "trbench.eval.queue_report --file results/gpu_6d_mnbt.jsonl",
    "trbench.eval.pareto",
]
OUT = Path("repro")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    failed = []
    for m in MODULES:
        name = m.split()[0].rsplit(".", 1)[1]
        r = subprocess.run([sys.executable, "-m", *m.split()], capture_output=True, text=True, check=False,
                           encoding="utf-8")
        (OUT / f"{name}.md").write_text(r.stdout, encoding="utf-8")
        status = "ok" if r.returncode == 0 else f"HATA ({r.returncode})"
        print(f"{status:10} {m}")
        if r.returncode:
            failed.append(m)
            print(r.stderr[-500:])
    print(f"\n{len(MODULES) - len(failed)}/{len(MODULES)} ok; ciktilar: {OUT}/")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
