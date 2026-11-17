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


def test_summary_md_one_row_per_model_with_fp16():
    from trbench.eval.analysis import summary_md
    rows = rows_for("fp16", 8, 9) + rows_for("q4_k_m", 5, 8)
    rows += [dict(r, label=r["label"].replace("m/", "n/")) for r in rows_for("q8_0", 5, 5)]  # n: fp16 yok
    md = summary_md(rows).splitlines()
    assert len(md) == 3 and md[2].startswith("| m |")
    assert "80 / 90" in md[2] and "+20.0" in md[2]


def test_clean_excludes_translation_errors():
    from trbench.eval.analysis import TRANSLATION_ERRORS, gap
    bad = "math/gsm8k-test-721"
    assert bad in TRANSLATION_ERRORS
    t = {(bad, "tr"): False, (bad, "en"): True, ("math/1", "tr"): True, ("math/1", "en"): True}
    assert gap(t, "math") == (0.5, 1.0)
    assert gap(t, "math", TRANSLATION_ERRORS) == (1.0, 1.0)
