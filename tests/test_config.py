from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from trbench.config import GPU2_PLAN_CAP_GB, SERVICE_RESERVE_GB, Phase3, load

CONFIGS = ["configs/phase3.yaml", "configs/gpu_vllm.yaml", "configs/gpu_llamacpp.yaml", "configs/gpu_probe_vllm.yaml",
           "configs/gpu_probe.yaml"]


def base(path="configs/gpu_vllm.yaml", **over):
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    d.update(over)
    return d


@pytest.mark.parametrize("path", CONFIGS)
def test_repo_configs_fit_gpu2_budget(path):
    cfg = load(path)
    for m in cfg.models:
        assert cfg.need_gb(m) + SERVICE_RESERVE_GB <= GPU2_PLAN_CAP_GB   # guard 17, alarm 20


@pytest.mark.parametrize("over,msg", [
    ({"port": 8001}, "18000-18099"),                     # baska bir servisin portu
    ({"gpu_memory_utilization": 0.80}, "servis alarmi"),  # 16 Kasim ayari
    ({"gpu_memory_utilization": 0.20, "enforce_eager": False}, "servis alarmi"),  # CUDA graph +4 GB
    ({"max_num_seqs": 2}, "max_num_seqs"),
])
def test_shared_server_rules_enforced(over, msg):
    with pytest.raises(ValidationError, match=msg):
        Phase3.model_validate(base(**over))


def test_too_big_model_rejected():
    big = {"name": "llama-3.1-8b", "quant": "fp16", "params_b": 8.0, "repo": "x"}  # 16 GB
    with pytest.raises(ValidationError, match="sigmaz"):
        Phase3.model_validate(base(models=[big]))


def test_llamacpp_kv_counts_toward_budget():
    gemma = {"name": "gemma-2-9b", "quant": "q4_k_m", "params_b": 9.2, "repo": "x",
             "weights_gb_actual": 5.8, "kv_kb_per_token": 344}           # 8 slot: KV ~4 GB
    with pytest.raises(ValidationError, match="sigmaz"):
        Phase3.model_validate(base("configs/gpu_llamacpp.yaml", models=[gemma]))
    ok = load("configs/gpu_llamacpp.yaml")
    g = next(m for m in ok.models if m.name == "gemma-2-9b")
    assert ok.conc(g) == [1, 2]


def test_runs_scale_requests_and_carry_engine():
    runs = load("configs/gpu_llamacpp.yaml").runs()
    assert {r["engine"] for r in runs} == {"llamacpp"}
    assert {r["n_requests"] for r in runs if r["concurrency"] == 8} == {32}
    assert {r["n_requests"] for r in runs if r["concurrency"] == 1} == {16}
