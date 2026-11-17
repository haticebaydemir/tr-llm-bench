"""FLORES+ paralel korpusunu indirir ve hizalar.

FLORES+ kapili bir veri setidir: metinler ASLA repo'ya veya yayina girmez.
Sadece yerel data/flores/ altina indirilir (.gitignore'da).

Kullanim:
  python -m trbench.tokenizer_tax.corpus
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = "openlanguagedata/flores_plus"
SPLITS = ("dev", "devtest")
# Turkce + Ingilizce + karsilastirma: iki Hint-Avrupa (deu, fra), iki eklemeli (fin, hun)
LANGS = ("eng_Latn", "tur_Latn", "deu_Latn", "fra_Latn", "fin_Latn", "hun_Latn")
DATA_DIR = Path("data/flores")


EXTS = (".jsonl", ".parquet")


def find_files(all_files: list[str], lang: str) -> dict[str, str]:
    """Repo dosya listesinden bir dilin dev/devtest dosyalarini bulur.

    Dosya adi dil koduyla TAM eslesmeli: 'eng_Latn' ararken 'ben_Beng' gelmesin.
    """
    out: dict[str, str] = {}
    for f in all_files:
        path = Path(f)
        if path.suffix not in EXTS or path.stem != lang:
            continue
        split = path.parts[0]
        if split in SPLITS and split not in out:
            out[split] = f
    return out


def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return pd.read_parquet(path)


def align(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Her dilin (id, text) tablosunu id uzerinden birlestirir.

    Sonuc: id + her dil icin bir sutun. Herhangi bir dilde eksik olan id atilir.
    """
    merged: pd.DataFrame | None = None
    for lang, df in frames.items():
        part = df[["id", "text"]].rename(columns={"text": lang})
        merged = part if merged is None else merged.merge(part, on="id", how="inner")
    if merged is None:
        raise ValueError("bos girdi")
    return merged.sort_values("id").reset_index(drop=True)


def download(langs: tuple[str, ...] = LANGS, out_dir: Path = DATA_DIR) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    frames: dict[str, pd.DataFrame] = {}
    for lang in langs:
        paths = find_files(files, lang)
        if set(paths) != set(SPLITS):
            raise RuntimeError(f"{lang}: dosyalar bulunamadi -> {paths}")
        parts = []
        for split in SPLITS:
            local = hf_hub_download(REPO, paths[split], repo_type="dataset")
            df = read_any(local)
            df["id"] = split + "_" + df["id"].astype(str)
            parts.append(df)
        frames[lang] = pd.concat(parts, ignore_index=True)
        print(f"{lang}: {len(frames[lang])} cumle")

    table = align(frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "flores_parallel.parquet"
    table.to_parquet(out, index=False)
    print(f"hizali: {len(table)} cumle x {len(langs)} dil -> {out}")
    return out


def main() -> None:
    argparse.ArgumentParser(description="FLORES+ indir ve hizala").parse_args()
    download()


if __name__ == "__main__":
    main()
