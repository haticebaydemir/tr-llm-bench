"""GSM8K TR/EN eslestirme denetimi.

Eslestirme final cevap dizisiyle yapiliyor (sources.pair_gsm8k). Ayni cevapli iki FARKLI
soru eslesebilir mi? Bagimsiz bir kontrol: ceviri sorudaki sayilari korur. TR ve EN
sorudaki sayi kumeleri ayniysa eslesme dogrudur; farkliysa elle bakilacak listeye girer.
(Sayi yaziyla yazilmissa -- 'iki' / 'two' -- yanlis alarm olabilir; o yuzden elle bakis.)

  python -m trbench.eval.audit_pairs            -> ozet + supheli ciftler
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from trbench.eval.build import OUT as EVAL

_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def norm(tok: str, lang: str) -> str:
    """'1.250' (TR binlik) / '1,250' (EN binlik) -> '1250'; '3,5' (TR) / '3.5' (EN) -> '3.5'."""
    thou, dec = (".", ",") if lang == "tr" else (",", ".")
    if re.fullmatch(rf"\d{{1,3}}(\{thou}\d{{3}})+", tok):
        return tok.replace(thou, "")
    return tok.replace(dec, ".")


def numbers(text: str, lang: str) -> set[str]:
    return {norm(t, lang) for t in _NUM.findall(text)}


def question(prompt: str) -> str:
    return prompt.split("Problem:", 1)[-1]


def audit(items: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for it in items:
        if it["category"] == "math":
            by_id.setdefault(it["id"], {})[it["lang"]] = it
    rows = []
    for i, d in sorted(by_id.items()):
        tr, en = numbers(question(d["tr"]["prompt"]), "tr"), numbers(question(d["en"]["prompt"]), "en")
        union = tr | en
        rows.append({"id": i, "jaccard": round(len(tr & en) / len(union), 2) if union else 1.0,
                     "only_tr": sorted(tr - en), "only_en": sorted(en - tr)})
    return rows


def main() -> None:
    items = [json.loads(x) for x in Path(EVAL).read_text(encoding="utf-8").splitlines()]
    rows = audit(items)
    exact = sum(r["jaccard"] == 1.0 for r in rows)
    print(f"{len(rows)} matematik cifti: {exact} tanesinde sayilar birebir ayni")
    for r in rows:
        if r["jaccard"] < 1.0:
            print(f"  {r['id']}: benzerlik {r['jaccard']}, sadece TR {r['only_tr']}, sadece EN {r['only_en']}")
    low = [r for r in rows if r["jaccard"] < 0.5]
    print(f"Ciddi supheli (benzerlik < 0.5): {len(low)}")


if __name__ == "__main__":
    main()
