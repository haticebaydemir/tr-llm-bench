from trbench.eval.analysis import BASE
from trbench.eval.stats import equivalence, holm, mcnemar_exact, pairwise


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # 10 uyusmayan ciftin hepsi tek tarafta: 2 * 0.5^10
    assert abs(mcnemar_exact(10, 0) - 2 / 1024) < 1e-12
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)


def test_holm_monotone_and_capped():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert abs(adj["a"] - 0.03) < 1e-12
    assert abs(adj["c"] - 0.06) < 1e-12
    assert adj["b"] >= adj["c"]
    assert all(v <= 1 for v in holm({"x": 0.9, "y": 0.8}).values())


def _tab(n, q_tr_wrong):
    ids = [f"reading/{i}" for i in range(n)]
    b = {(i, lg): True for i in ids for lg in ("tr", "en")}
    q = dict(b)
    for i in ids[:q_tr_wrong]:
        q[(i, "tr")] = False
    return {BASE: b, "q4": q}


def test_equivalence_identical_has_zero_margin():
    r = equivalence(_tab(50, 0), "q4", n=200)
    assert r["tax"]["est"] == 0 and r["tax"]["min_equiv_margin"] == 0
    assert r["loss_tr"]["max_excluded_loss"] == 0


def test_equivalence_detects_tr_loss():
    r = equivalence(_tab(100, 20), "q4", n=500)
    assert abs(r["loss_tr"]["est"] - 20) < 1e-9
    assert r["tax"]["ci95"][0] > 0
    assert r["loss_tr"]["max_excluded_loss"] >= 20


def test_pairwise_tiers():
    ids = [str(i) for i in range(100)]
    ans = {"good": {i: True for i in ids},
           "good2": {i: int(i) >= 2 for i in ids},
           "bad": {i: int(i) >= 60 for i in ids}}
    res = pairwise(ans)
    assert res["n_items"] == 100
    assert res["tier"]["good"] == res["tier"]["good2"] == 1
    assert res["tier"]["bad"] == 2
