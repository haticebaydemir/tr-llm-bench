from trbench.eval.contamination import split_of, table


def _row(model, cat, lang, ok, i):
    return {"label": f"{model}/q4_k_m", "category": cat, "lang": lang, "correct": ok,
            "id": f"{cat}/{i}"}


def test_split_of():
    assert split_of("reading") == "public" and split_of("json") == "original"
    assert split_of("morphology") is None


def test_table_gaps():
    rows = []
    for i in range(4):
        rows += [_row("m", "math", "tr", i < 2, i), _row("m", "math", "en", True, i),
                 _row("m", "json", "tr", True, i), _row("m", "json", "en", True, i),
                 _row("m", "morphology", "tr", False, i)]
    w = table(rows)
    assert w.loc["m", "public_tr"] == 0.5 and w.loc["m", "public_gap"] == -0.5
    assert w.loc["m", "original_gap"] == 0
