"""Hazir veri setlerinden eval sorulari secer.

Kural: secim deterministik (sabit seed) -> herkes ayni soruyu alir.
"""

from __future__ import annotations

import random
import re

import pandas as pd

from trbench.eval.schema import Item

SEED = 2026
LETTERS = "ABCD"

PROMPT = {
    "tr": ("Aşağıdaki metni okuyun ve soruyu cevaplayın. "
           "Sadece doğru seçeneğin harfini yazın (A, B, C veya D).\n\n"
           "Metin: {passage}\n\nSoru: {question}\n\n{options}\n\nCevap:"),
    "en": ("Read the passage below and answer the question. "
           "Write only the letter of the correct option (A, B, C or D).\n\n"
           "Passage: {passage}\n\nQuestion: {question}\n\n{options}\n\nAnswer:"),
}


def belebele_items(df: pd.DataFrame, lang: str, ids: list[str]) -> list[Item]:
    """Belebele satirlarini Item'a cevirir. id = link + soru numarasi (dillerde ortak)."""
    df = df.assign(id=df["link"].astype(str) + "#" + df["question_number"].astype(str))
    rows = df.set_index("id").loc[ids]
    items = []
    for item_id, r in rows.iterrows():
        options = "\n".join(f"{LETTERS[i]}) {r[f'mc_answer{i + 1}']}" for i in range(4))
        items.append(Item(
            id=f"reading/{item_id}",
            category="reading",
            lang=lang,
            prompt=PROMPT[lang].format(passage=r["flores_passage"], question=r["question"],
                                       options=options),
            answer=LETTERS[int(r["correct_answer_num"]) - 1],
            scorer="choice",
            source="facebook/belebele (CC BY-SA 4.0)",
        ))
    return items


def pick_parallel_ids(tr: pd.DataFrame, en: pd.DataFrame, n: int) -> list[str]:
    """Iki dilde de bulunan id'lerden n tanesini sabit seed ile secer."""
    def ids(df: pd.DataFrame) -> set[str]:
        return set(df["link"].astype(str) + "#" + df["question_number"].astype(str))

    common = sorted(ids(tr) & ids(en))
    if len(common) < n:
        raise ValueError(f"yeterli paralel soru yok: {len(common)} < {n}")
    return sorted(random.Random(SEED).sample(common, n))


# ---------- Matematik: GSM8K (EN) <-> GSM8K-TR v0.2 ----------

# Dusunme ZORUNLU. Ilk surumde "dusunebilirsiniz ... yalnizca Cevap yazin" denmisti;
# model bazen hic dusunmeden sadece "Cevap: 66" yazdi ve bu davranis dile/quant'a gore
# degisti (Q4 EN %20, TR %70; Q8 TR %23). Olculen sey dil degil, "dusundu mu" oldu.
MATH_PROMPT = {
    "tr": ("Aşağıdaki problemi önce adım adım çözün. Çözümünüzü yazdıktan sonra "
           "en son satıra \"Cevap: <sayı>\" yazın.\n\nProblem: {q}"),
    "en": ("Solve the problem below step by step first. After writing your solution, "
           "put \"Answer: <number>\" on the very last line.\n\nProblem: {q}"),
}
_TR_THOUSANDS = re.compile(r"^\d{1,3}(\.\d{3})+$")


def final_number(answer: str) -> str | None:
    """GSM8K cevabindaki '#### 1,250' kismindan sayiyi normalize eder -> '1250'.

    TR ceviride binlik ayirici nokta olabilir ('1.250') -> o da '1250'.
    """
    if "####" not in answer:
        return None
    raw = answer.split("####")[-1].strip().replace(" ", "")
    if _TR_THOUSANDS.match(raw):
        raw = raw.replace(".", "")
    raw = raw.replace(",", "")
    return raw or None


MIN_BLOCK = 3  # bu kadar ardisik eslesmeyen kisa bloklar tesaduf sayilir


def pair_gsm8k(tr: pd.DataFrame, en: pd.DataFrame) -> list[tuple[int, int]]:
    """TR ve EN satirlarini final cevap dizisi uzerinden hizalar -> [(tr_idx, en_idx)].

    Ceviride satir silinmis/eklenmis olabilir (gercekte: tr=1317, en=1319).
    Sira-sira eslestirmek, kaymadan sonraki HER soruyu yanlis cevapla eslestirirdi.
    Bunun yerine 'diff' mantigi: iki cevap dizisinin ortak bloklarini bulur.
    """
    from difflib import SequenceMatcher

    a = [final_number(x) for x in tr["answer"]]
    b = [final_number(x) for x in en["answer"]]
    pairs = []
    for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        if blk.size < MIN_BLOCK:
            continue
        for k in range(blk.size):
            if a[blk.a + k] is not None:
                pairs.append((blk.a + k, blk.b + k))
    return pairs


def gsm8k_items(tr: pd.DataFrame, en: pd.DataFrame, n: int) -> list[Item]:
    pairs = pair_gsm8k(tr, en)
    if len(pairs) < n:
        raise ValueError(f"yeterli eslesen soru yok: {len(pairs)} < {n}")
    chosen = sorted(random.Random(SEED).sample(pairs, n))
    rate = round(len(pairs) / len(en), 4)
    items = []
    for lang, src in (("tr", "malhajar/gsm8k_tr-v0.2"), ("en", "openai/gsm8k (MIT)")):
        for ti, ei in chosen:
            df, i = (tr, ti) if lang == "tr" else (en, ei)
            items.append(Item(
                id=f"math/gsm8k-test-{ei}",          # id hep orijinal EN indeksi
                category="math",
                lang=lang,
                prompt=MATH_PROMPT[lang].format(q=df.iloc[i]["question"]),
                answer=final_number(df.iloc[i]["answer"]),
                scorer="number",
                source=src,
                meta={"match_rate": rate},
            ))
    return items


# ---------- Bilgi: Global-MMLU (CohereLabs, Apache 2.0) ----------

MMLU_PROMPT = {
    "tr": ("Aşağıdaki soruyu cevaplayın. Sadece doğru seçeneğin harfini yazın "
           "(A, B, C veya D).\n\nSoru: {question}\n\n{options}\n\nCevap:"),
    "en": ("Answer the question below. Write only the letter of the correct option "
           "(A, B, C or D).\n\nQuestion: {question}\n\n{options}\n\nAnswer:"),
}
MMLU_COLS = ("sample_id", "question", "option_a", "option_b", "option_c", "option_d", "answer",
             "cultural_sensitivity_label")   # yoksa CA filtresi yapilamaz -> hata, sessizce hepsini alma


def global_mmlu_items(tr: pd.DataFrame, en: pd.DataFrame, n: int) -> list[Item]:
    """sample_id uzerinden paralel; sadece kulturel olarak NOTR (CA) sorular.

    'CS' (culturally sensitive) sorular bir kulture ozgu bilgi ister (ABD tarihi gibi).
    TR/EN kiyasinda dil etkisini olcmek istiyoruz, kultur etkisini degil -> CA.
    """
    for name, df in (("tr", tr), ("en", en)):
        missing = [c for c in MMLU_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Global-MMLU {name}: eksik sutun {missing}; var: {list(df.columns)}")

    def ca(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["cultural_sensitivity_label"] == "CA"]

    common = sorted(set(ca(tr)["sample_id"]) & set(ca(en)["sample_id"]))
    if len(common) < n:
        raise ValueError(f"yeterli paralel soru yok: {len(common)} < {n}")
    chosen = sorted(random.Random(SEED).sample(common, n))

    items = []
    for lang, df in (("tr", tr), ("en", en)):
        rows = df.drop_duplicates("sample_id").set_index("sample_id").loc[chosen]
        for sid, r in rows.iterrows():
            options = "\n".join(f"{L}) {r[f'option_{L.lower()}']}" for L in LETTERS)
            items.append(Item(
                id=f"knowledge/{sid}",
                category="knowledge",
                lang=lang,
                prompt=MMLU_PROMPT[lang].format(question=r["question"], options=options),
                answer=str(r["answer"]).strip().upper(),
                scorer="choice",
                source="CohereLabs/Global-MMLU (Apache 2.0)",
                meta={"subject": r.get("subject", "")},
            ))
    return items
