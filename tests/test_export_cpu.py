from trbench.eval.export_cpu import long_text_fields, strip


def test_strip_drops_response_only():
    r = {"label": "m/fp16", "id": "reading/x", "correct": True, "response": "A long answer", "raw_response": "raw"}
    assert strip(r) == {"label": "m/fp16", "id": "reading/x", "correct": True}


def test_long_text_fields():
    assert long_text_fields({"id": "x" * 100, "platform": "y" * 81}) == ["platform"]
