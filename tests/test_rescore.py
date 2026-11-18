from trbench.eval.rescore import cut, rescore
from trbench.eval.schema import Item


def test_cut_at_first_turn_end():
    assert cut("\nB) So the image\n\n<|im_end|>") == "\nB) So the image\n\n"
    assert cut("<|im_end|>\n\nAnkara'da\n") == ""          # bos cevap -> yanlis olmali
    assert cut("plain answer") == "plain answer"


def test_rescore_only_target_model_and_idempotent():
    it = Item(id="morph/1", category="morphology", lang="tr", prompt="p",
              answer="Ankara'da", scorer="morph", source="x")
    rows = [{"label": "trendyol-7b/fp16", "id": "morph/1", "lang": "tr",
             "response": "<|im_end|>\n\nAnkara'da\n", "correct": True},
            {"label": "qwen2.5-7b/fp16", "id": "morph/1", "lang": "tr",
             "response": "Ankara'da", "correct": True}]
    items = {("morph/1", "tr"): it}
    assert rescore(rows, "trendyol-7b", items) == 1
    assert rows[0]["correct"] is False and rows[0]["raw_response"].startswith("<|im_end|>")
    assert rows[1] == {"label": "qwen2.5-7b/fp16", "id": "morph/1", "lang": "tr",
                       "response": "Ankara'da", "correct": True}
    assert rescore(rows, "trendyol-7b", items) == 0      # ikinci kez: degisiklik yok
