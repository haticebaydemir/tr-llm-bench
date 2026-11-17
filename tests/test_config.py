from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from trbench.config import Phase3, load


def base(**over):
    d = yaml.safe_load(Path("configs/phase3.yaml").read_text(encoding="utf-8"))
    d.update(over)
    return d


def test_repo_config_is_valid():
    cfg = load("configs/phase3.yaml")
    assert len(cfg.runs()) == len(cfg.models) * 2 * 4 * 3
    assert all(m.quant.endswith("int4") for m in cfg.models)   # GPU 2'ye sadece 4-bit sigar


@pytest.mark.parametrize("over,msg", [
    ({"port": 8001}, "18000-18099"),                     # baska bir servisin portu
    ({"gpu_memory_utilization": 0.9}, "servis alarmi"),
    ({"gpu_memory_utilization": 0.65}, "servis alarmi"),          # 16 Kasim olayindan sonra reddedilir
    ({"max_num_seqs": 16}, "max_num_seqs"),
])
def test_shared_server_rules_enforced(over, msg):
    with pytest.raises(ValidationError, match=msg):
        Phase3.model_validate(base(**over))


def test_too_big_model_rejected():
    big = {"name": "llama-3.1-8b", "quant": "fp16", "params_b": 8.0, "repo": "x"}  # 16 GB
    with pytest.raises(ValidationError, match="sigmaz"):
        Phase3.model_validate(base(models=[big]))


def test_runs_scale_requests_with_concurrency():
    runs = load("configs/phase3.yaml").runs()
    assert {r["n_requests"] for r in runs if r["concurrency"] == 32} == {128}
    assert {r["n_requests"] for r in runs if r["concurrency"] == 1} == {16}
