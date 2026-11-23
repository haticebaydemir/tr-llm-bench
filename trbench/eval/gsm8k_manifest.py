"""Faz 7i: GSM8K ciftlerine soru ozeti ve veri seti surumu ekler (bir kez, yerelde).

  python -m trbench.eval.gsm8k_manifest

Ciftler degismez; yalnizca her cift icin TR/EN soru metninin ozeti ve iki veri setinin
commit sha'si yazilir. Sonra build bu surumu indirir ve ozetler tutmazsa durur.
"""

from __future__ import annotations

import json

from huggingface_hub import HfApi

from trbench.eval.build import N, load_gsm8k
from trbench.eval.sources import GSM8K_PAIRS, GSM8K_REPOS, add_hashes, choose_gsm8k, load_manifest


def main() -> None:
    doc = load_manifest()
    if doc is None:
        raise SystemExit(f"{GSM8K_PAIRS} yok: once python -m trbench.eval.build")
    revs = {lang: HfApi().dataset_info(repo).sha for lang, repo in GSM8K_REPOS.items()}
    tr, en = load_gsm8k(revs)
    choose_gsm8k(tr, en, N["math"], [(p["tr"], p["en"]) for p in doc["pairs"]])   # ciftler hala gecerli mi
    new = add_hashes(doc, tr, en, revs)
    GSM8K_PAIRS.write_text(json.dumps(new, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"yazildi {GSM8K_PAIRS}: {len(new['pairs'])} cift, surumler {revs}")


if __name__ == "__main__":
    main()
