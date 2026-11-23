

def test_pick_uses_timestamp_not_file_order():
    from trbench.results import pick
    new = {"m": "a", "q": 1, "exp": "yeni", "ts": "2026-11-23T10:00:00+00:00", "v": 2}
    old = {"m": "a", "q": 1, "exp": "eski", "ts": "2026-11-20T10:00:00+00:00", "v": 1}
    assert [r["v"] for r in pick([new, old], ("m",), ("q",), warn=lambda _: None)] == [2]


def test_pick_duplicate_item_keeps_newest_ts():
    from trbench.results import pick
    a = {"m": "a", "q": 1, "exp": "e", "ts": "2026-11-23T10:00:00+00:00", "v": "yeni"}
    b = {"m": "a", "q": 1, "exp": "e", "ts": "2026-11-22T10:00:00+00:00", "v": "eski"}
    assert [r["v"] for r in pick([a, b], ("m",), ("q",), warn=lambda _: None)] == ["yeni"]


def test_pick_without_ts_falls_back_to_file_order():
    from trbench.results import pick
    rows = [{"m": "a", "q": 1, "exp": "x", "v": 1}, {"m": "a", "q": 1, "exp": "y", "v": 2}]
    assert [r["v"] for r in pick(rows, ("m",), ("q",), warn=lambda _: None)] == [2]
