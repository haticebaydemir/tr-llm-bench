from trbench.guard import GuardConfig, Snapshot, check, parse_nvidia_smi, preflight

CFG = GuardConfig()


def snap(**kw) -> Snapshot:
    """16 Kasim 2026 sunucu degerleriyle saglikli bir anlik goruntu."""
    base = {
        "gpu_used_mib": {0: 29440, 1: 29132, 2: 1550, 3: 25028},
        "gpu_free_mib": {0: 3055, 1: 3363, 2: 30945, 3: 7467},
        "root_free_gb": 118.0,
        "disk3_used_pct": 3.0,
        "service_ok": True,
        "service_latency_ms": 20.0,
    }
    base.update(kw)
    return Snapshot(**base)


def test_healthy_no_violation():
    assert check(snap(), snap(), CFG) == []


def test_bench_gpu_usage_within_cap_is_allowed():
    # 7B AWQ + servis: ~15.7 GB -> sorun yok
    now = snap(gpu_used_mib={0: 29440, 1: 29132, 2: 15700, 3: 25028},
               gpu_free_mib={0: 3055, 1: 3363, 2: 16800, 3: 7467})
    assert check(snap(), now, CFG) == []


def test_real_incident_gpu2_total_over_service_alarm():
    # 16 Kasim 20:05: GPU 2 toplam 27428 MiB -> servisin alarmi calmisti
    now = snap(gpu_used_mib={0: 29440, 1: 29132, 2: 27428, 3: 25028},
               gpu_free_mib={0: 3055, 1: 3363, 2: 5000, 3: 7467})
    assert any("servis alarmi" in x for x in check(snap(), now, CFG))


def test_protected_gpu_change_detected():
    now = snap(gpu_used_mib={0: 31500, 1: 29132, 2: 1550, 3: 25028})
    v = check(snap(), now, CFG)
    assert len(v) == 1 and "GPU 0" in v[0]


def test_small_fluctuation_tolerated():
    now = snap(gpu_used_mib={0: 29440 + 500, 1: 29132, 2: 1550, 3: 25028})
    assert check(snap(), now, CFG) == []


def test_missing_gpu_detected():
    now = snap(gpu_used_mib={0: 29440, 1: 29132, 2: 1550})
    assert any("GPU 3 okunamadi" in x for x in check(snap(), now, CFG))


def test_bench_gpu_squeezing_service():
    now = snap(gpu_free_mib={0: 3055, 1: 3363, 2: 500, 3: 7467})
    assert any("GPU 2" in x for x in check(snap(), now, CFG))


def test_disk_limits():
    v = check(snap(), snap(root_free_gb=90.0, disk3_used_pct=86.0), CFG)
    assert len(v) == 2


def test_service_down():
    v = check(snap(), snap(service_ok=False, service_latency_ms=None), CFG)
    assert v == ["Servis yanit vermiyor"]


def test_latency_floor_prevents_false_alarm():
    # 20 ms -> 100 ms: 3 katindan fazla ama 500 ms tabaninin altinda -> alarm yok
    assert check(snap(), snap(service_latency_ms=100.0), CFG) == []


def test_latency_spike_detected():
    assert any("gecikme" in x for x in check(snap(), snap(service_latency_ms=900.0), CFG))


def test_preflight_ok_today():
    assert preflight(snap(), CFG) == []


def test_preflight_blocks_when_gpu2_busy():
    now = snap(gpu_used_mib={0: 29440, 1: 29132, 2: 9000, 3: 25032})
    assert len(preflight(now, CFG)) == 1


def test_preflight_ok_with_service_model_loaded():
    now = snap(gpu_used_mib={0: 29440, 1: 29132, 2: 3664, 3: 25032})
    assert preflight(now, CFG) == []


def test_parse_nvidia_smi():
    used, free = parse_nvidia_smi("0, 29440, 3055\n1, 29132, 3363\n2, 1550, 30945\n")
    assert used == {0: 29440, 1: 29132, 2: 1550}
    assert free[2] == 30945


def test_gpu3_scraper_fluctuation_tolerated_but_gpu0_not():
    # 16 Kasim gercek olay: GPU 3 25032 -> 26548 (scraper), yanlis alarmdi
    now = snap(gpu_used_mib={0: 29440, 1: 29132, 2: 1550, 3: 26548})
    assert check(snap(), now, CFG) == []
    now = snap(gpu_used_mib={0: 29440 + 1500, 1: 29132, 2: 1550, 3: 25028})
    assert any("GPU 0" in x for x in check(snap(), now, CFG))


def test_real_incident_gpu2_767mib_is_violation():
    now = snap(gpu_free_mib={0: 3055, 1: 3363, 2: 767, 3: 7467})
    assert any("GPU 2" in x for x in check(snap(), now, CFG))
