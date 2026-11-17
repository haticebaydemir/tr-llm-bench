"""Eval setini kaynaklardan uretir -> data/eval/eval_set.jsonl

Kullanim:
  python -m trbench.eval.build
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trbench.eval.ifeval import ifeval_items
from trbench.eval.json_tasks import json_items
from trbench.eval.morphology import morphology_items
from trbench.eval.schema import Item
from trbench.eval.sources import (
    belebele_items,
    global_mmlu_items,
    gsm8k_items,
    pick_parallel_ids,
)

OUT = Path("data/eval/eval_set.jsonl")
N = {"reading": 60, "math": 40, "knowledge": 40, "instruction": 60, "json": 50, "morphology": 40}


def load_belebele(code: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("facebook/belebele", f"data/{code}.jsonl", repo_type="dataset")
    return pd.read_json(path, lines=True)


def build_reading() -> list[Item]:
    tr, en = load_belebele("tur_Latn"), load_belebele("eng_Latn")
    ids = pick_parallel_ids(tr, en, N["reading"])
    return belebele_items(tr, "tr", ids) + belebele_items(en, "en", ids)


def load_test_parquet(repo: str, must_contain: str = "") -> pd.DataFrame:
    """Repo'daki test parquet dosyasini bulur (yolu sabit yazmadan)."""
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
             if f.endswith(".parquet") and "test" in f and must_contain in f]
    if len(files) != 1:
        raise RuntimeError(f"{repo}: test dosyasi belirsiz -> {files}")
    return pd.read_parquet(hf_hub_download(repo, files[0], repo_type="dataset"))


def build_math() -> list[Item]:
    en = load_test_parquet("openai/gsm8k", must_contain="main/")
    tr = load_test_parquet("malhajar/gsm8k_tr-v0.2")
    items = gsm8k_items(tr, en, N["math"])
    print(f"GSM8K eslesme orani: %{items[0].meta['match_rate'] * 100:.1f} "
          f"({len(tr)} satir)")
    return items


def build_knowledge() -> list[Item]:
    tr = load_test_parquet("CohereLabs/Global-MMLU", must_contain="tr/")
    en = load_test_parquet("CohereLabs/Global-MMLU", must_contain="en/")
    return global_mmlu_items(tr, en, N["knowledge"])


def write(items: list[Item], out: Path = OUT) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    items = (build_reading() + build_math() + build_knowledge()
             + ifeval_items(N["instruction"]) + json_items(N["json"]) + morphology_items(N["morphology"]))
    write(items)
    by = pd.DataFrame([i.to_dict() for i in items]).groupby(["category", "lang"]).size()
    print(by.to_string())
    print(f"\nkaydedildi: {OUT} ({len(items)} soru)")
    for cat in ("morphology",):
        tr0 = next(i for i in items if i.lang == "tr" and i.category == cat)
        print(f"\nORNEK {cat} (TR):\n{tr0.prompt[-350:]}\n-> dogru cevap: {tr0.answer}")


if __name__ == "__main__":
    main()
