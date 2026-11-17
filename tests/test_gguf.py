from trbench.eval.gguf import first_part, pick_files

FILES = [
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
    "qwen2.5-7b-instruct-q4_k_s.gguf",                 # q4_k_m ile karismamali
    "qwen2.5-7b-instruct-q8_0.gguf",
    "qwen2.5-7b-instruct-fp16-00001-of-00004.gguf",
    "README.md",
]


def test_pick_split_and_not_similar_quant():
    assert pick_files(FILES, "q4_k_m") == FILES[:2]
    assert pick_files(FILES, "Q8_0") == ["qwen2.5-7b-instruct-q8_0.gguf"]
    assert pick_files(FILES, "q5_k_m") == []


def test_first_part():
    assert first_part(pick_files(FILES, "q4_k_m")).endswith("00001-of-00002.gguf")
    assert first_part(["a-q8_0.gguf"]) == "a-q8_0.gguf"
