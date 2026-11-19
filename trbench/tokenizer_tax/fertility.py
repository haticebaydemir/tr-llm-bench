"""Tokenizer vergisi: ayni anlamdaki metin her dilde kac token tutuyor?

Iki metrik:
  - fertility   = token / kelime  (dil ici; diller arasi kiyas icin YANILTICI)
  - tax_vs_eng  = dildeki toplam token / Ingilizcedeki toplam token
                  (paralel korpus -> icerik ayni -> asil vergi bu)

Kullanim:
  python -m trbench.tokenizer_tax.fertility
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from trbench.metrics.text import count_words
from trbench.tokenizer_tax.corpus import DATA_DIR, LANGS

# etiket -> Hugging Face repo (sadece tokenizer dosyalari indirilir, model agirligi degil)
MODELS = {
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    # Meta onayi beklenirken: unsloth'un acik kopyasi, tokenizer orijinalle ayni.
    # Onay gelince orijinalle dogrulanacak.
    "Llama-3.1-8B": "unsloth/Llama-3.1-8B-Instruct",
    "Gemma-2-9B": "google/gemma-2-9b-it",
    "Mistral-7B-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Mistral-Nemo-12B": "mistralai/Mistral-Nemo-Instruct-2407",
    "Cosmos-TR-Llama-8B": "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1",
    "Trendyol-7B": "Trendyol/Trendyol-LLM-7b-chat-v1.8",
    "Gemma-3-12B": "google/gemma-3-12b-it",
    "Aya-Expanse-8B": "CohereLabs/aya-expanse-8b",
    "gpt-oss-20B": "openai/gpt-oss-20b",
    "Phi-4": "microsoft/phi-4",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    # Yeni nesil ve cok dilli modeller (19 Kasim eki; gated/silinmis olan atlanir, raporlanir)
    "Ministral-8B": "mistralai/Ministral-8B-Instruct-2410",
    "Phi-4-mini": "microsoft/Phi-4-mini-instruct",
    "Granite-3.3-8B": "ibm-granite/granite-3.3-8b-instruct",
    "EuroLLM-9B": "utter-project/EuroLLM-9B-Instruct",
    "Kumru-2B": "vngrs-ai/Kumru-2B",
}
REF = "eng_Latn"
# Windows'ta symlink uyarisi her modelde tekrarlaniyor; zararsiz, gizle.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
OUT = Path("results/tokenizer_tax.csv")
# Cumle basina token sayilari (metin YOK, sadece sayi) -> guven araligi icin (langs.py)
COUNTS = Path("results/tokenizer_counts.csv.gz")

TokenizeFn = Callable[[list[str]], Iterable[int]]


def sentence_counts(texts: dict[str, list[str]], count_tokens: TokenizeFn) -> pd.DataFrame:
    """Uzun tablo: lang, sent (cumle sirasi), tokens. Paralel korpus: ayni sent = ayni anlam."""
    return pd.concat([pd.DataFrame({"lang": lang, "sent": range(len(s)), "tokens": list(count_tokens(s))})
                      for lang, s in texts.items()], ignore_index=True)


def compute_stats(texts: dict[str, list[str]], count_tokens: TokenizeFn) -> pd.DataFrame:
    """Saf fonksiyon: her dil icin toplam token/kelime/karakter ve oranlar."""
    rows = []
    for lang, sents in texts.items():
        tokens = sum(count_tokens(sents))
        words = sum(count_words(s) for s in sents)
        chars = sum(len(s) for s in sents)
        rows.append({"lang": lang, "tokens": tokens, "words": words, "chars": chars,
                     "fertility": tokens / words, "chars_per_token": chars / tokens})
    df = pd.DataFrame(rows)
    ref_tokens = df.loc[df["lang"] == REF, "tokens"]
    df["tax_vs_eng"] = df["tokens"] / ref_tokens.iloc[0] if len(ref_tokens) else float("nan")
    return df


# Bu tokenizer'larin HF'deki regex'i hatali; transformers duzeltme bayragi istiyor.
NEEDS_MISTRAL_REGEX_FIX = ("mistralai/Mistral-Nemo-Instruct-2407",)


def load_tokenizer(repo: str, revision: str | None = None):
    from transformers import AutoTokenizer

    kwargs = {"fix_mistral_regex": True} if repo in NEEDS_MISTRAL_REGEX_FIX else {}
    return AutoTokenizer.from_pretrained(repo, revision=revision, **kwargs)


def repo_revision(repo: str) -> str | None:
    """Repo'nun su anki commit'i: tokenizer dosyalari degisirse sonuc da degisir -> kaydet ve sabitle."""
    from huggingface_hub import HfApi

    try:
        return HfApi().model_info(repo).sha
    except Exception:  # noqa: BLE001
        return None


def lib_versions() -> dict:
    import tokenizers
    import transformers

    return {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__}


def hf_counter(tok) -> TokenizeFn:
    def count(sents: list[str]) -> list[int]:
        # add_special_tokens=False: BOS/EOS ve chat sablonu sayilmaz, sadece metnin kendisi
        enc = tok(sents, add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in enc]

    return count


def main() -> None:
    table = pd.read_parquet(DATA_DIR / "flores_parallel.parquet")
    texts = {lang: table[lang].tolist() for lang in LANGS}

    results, counts, skipped = [], [], []
    libs = lib_versions()
    for label, repo in MODELS.items():
        try:
            rev = repo_revision(repo)
            tok = load_tokenizer(repo, rev)
            df = compute_stats(texts, hf_counter(tok))
            cnt = sentence_counts(texts, hf_counter(tok))
        except Exception as e:  # noqa: BLE001 -- gated, silinmis repo vb. -> atla, raporla
            skipped.append(f"{label}: {type(e).__name__}: {str(e).splitlines()[0][:90]}")
            continue
        df.insert(0, "model", label)
        df.insert(1, "repo", repo)
        df.insert(2, "vocab_size", len(tok))
        df.insert(3, "revision", rev)
        df.insert(4, "tokenizer_class", type(tok).__name__)
        for k, v in libs.items():
            df[k] = v
        results.append(df)
        counts.append(cnt.assign(model=label))
        tr = df.set_index("lang").loc["tur_Latn"]
        print(f"{label:20s} vocab {len(tok):>7,} | TR fertility {tr.fertility:.2f}"
              f" | TR/EN token {tr.tax_vs_eng:.2f}x")

    if skipped:
        print("\nATLANAN:")
        for s in skipped:
            print("  -", s)
    if not results:
        raise SystemExit("hicbir tokenizer yuklenemedi")

    out = pd.concat(results, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    pd.concat(counts, ignore_index=True).to_csv(COUNTS, index=False)
    print(f"\nkaydedildi: {OUT}")
    pivot = out.pivot(index="model", columns="lang", values="tax_vs_eng").round(2)
    print("\nToken orani (dil / Ingilizce):")
    print(pivot[list(LANGS)].sort_values("tur_Latn").to_string())


if __name__ == "__main__":
    main()
