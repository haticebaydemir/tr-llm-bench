from trbench.results import pick


def _r(label, i, exp, ok=True):
    return {"label": label, "id": i, "lang": "tr", "exp": exp, "correct": ok}


def test_legacy_rows_without_exp_unchanged():
    rows = [_r("a", "1", None), _r("a", "2", None), _r("b", "1", None)]
    msgs = []
    assert pick(rows, ("label",), ("id", "lang"), msgs.append) == rows and msgs == []


def test_latest_experiment_wins_with_warning():
    rows = [_r("a", "1", "eski", False), _r("a", "2", "eski"), _r("a", "1", "yeni"), _r("b", "1", "x")]
    msgs = []
    out = pick(rows, ("label",), ("id", "lang"), msgs.append)
    assert out == [_r("a", "1", "yeni"), _r("b", "1", "x")]
    assert len(msgs) == 1 and "a" in msgs[0]


def test_duplicate_item_keeps_last_with_warning():
    rows = [_r("a", "1", "e", False), _r("a", "1", "e", True)]
    msgs = []
    assert pick(rows, ("label",), ("id", "lang"), msgs.append) == [_r("a", "1", "e", True)]
    assert len(msgs) == 1
