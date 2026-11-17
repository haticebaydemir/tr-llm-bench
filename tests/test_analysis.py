from trbench.eval.analysis import bootstrap_ci, paired_ids, table, tax, tr_wrong_en_right


def rows_for(quant, tr_correct, en_correct, n=10):
    out = []
    for k in range(n):
        out.append({"label": f"m/{quant}", "id": f"math/{k}", "lang": "tr",
                    "correct": k < tr_correct, "error": None})
        out.append({"label": f"m/{quant}", "id": f"math/{k}", "lang": "en",
                    "correct": k < en_correct, "error": None})
    return out


def test_tax_sign_and_value():
    rows = rows_for("fp16", 8, 9) + rows_for("q4_k_m", 5, 8)   # TR -3, EN -1
    tab = table(rows, "m")
    ids = paired_ids(tab, "q4_k_m")
    t = tax(tab, "q4_k_m", ids)
    assert round(t["loss_tr"], 2) == 0.3 and round(t["loss_en"], 2) == 0.1
    assert round(t["tax"], 2) == 0.2                            # TR 2 puan/10 fazla kaybediyor


def test_paired_ids_require_all_four_cells():
    rows = rows_for("fp16", 5, 5) + rows_for("q8_0", 5, 5, n=6)  # q8'de sadece 6 soru
    tab = table(rows, "m")
    assert len(paired_ids(tab, "q8_0")) == 6


def test_bootstrap_ci_contains_point_estimate():
    rows = rows_for("fp16", 8, 9, n=40) + rows_for("q4_k_m", 5, 8, n=40)
    tab = table(rows, "m")
    ids = paired_ids(tab, "q4_k_m")
    lo, hi = bootstrap_ci(tab, "q4_k_m", ids, n=500)
    assert lo <= tax(tab, "q4_k_m", ids)["tax"] <= hi


def test_tr_wrong_en_right():
    rows = rows_for("fp16", 3, 6)
    assert tr_wrong_en_right(rows, "m/fp16") == ["math/3", "math/4", "math/5"]
