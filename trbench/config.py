"""Deney config'i: YAML -> dogrulanmis Python nesnesi -> calistirilacak kosu listesi.

Paylasilan sunucu kurallari burada KOD olarak uygulanir: ihlal eden config hic calismaz.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

GPU_GB = 32.0
BYTES_PER_PARAM = {"fp16": 2.0, "gptq-int8": 1.0, "awq-int4": 0.5, "gptq-int4": 0.5}
# vLLM'in gpu_memory_utilization DISINDA kullandigi bellek (CUDA context + CUDA graph).
# 16 Kasim olcumu: 0.80 ayarinda kartin ~%86'si kullanildi -> ~4 GB.
OVERHEAD_GB = 4.0
# GPU 2 toplam tavani: servisin alarmi 20 GB'da, guard 17 GB'da durdurur; plan 16 GB'i asmaz.
GPU2_PLAN_CAP_GB = 16.0
# GPU 2'yi paylasan servisin kullanimi (olculen, modeli yukluyken) -> 4 GB
SERVICE_RESERVE_GB = 4.0
MIN_KV_GB = 2.0            # KV-cache icin en az bu kadar yer kalmali
MAX_UTIL = (GPU2_PLAN_CAP_GB - OVERHEAD_GB - SERVICE_RESERVE_GB) / GPU_GB   # 0.25


class ModelSpec(BaseModel):
    name: str
    quant: Literal["fp16", "gptq-int8", "awq-int4", "gptq-int4"]
    params_b: float = Field(gt=0)
    repo: str

    def weights_gb(self) -> float:
        return self.params_b * BYTES_PER_PARAM[self.quant]


class Phase3(BaseModel):
    engine: Literal["vllm"]
    port: int
    gpu_memory_utilization: float
    max_model_len: int = Field(ge=512)
    max_tokens: int = Field(ge=16)
    max_num_seqs: int = Field(ge=1)
    concurrency: list[int]
    langs: list[Literal["tr", "en"]]
    repeats: int = Field(ge=1)
    models: list[ModelSpec]

    @field_validator("port")
    @classmethod
    def port_in_our_range(cls, v: int) -> int:
        if not 18000 <= v <= 18099:
            raise ValueError("port 18000-18099 disinda (kendi port araligimiz)")
        return v

    @field_validator("gpu_memory_utilization")
    @classmethod
    def util_cap(cls, v: float) -> float:
        if not 0 < v <= min(0.80, MAX_UTIL):
            raise ValueError(f"gpu_memory_utilization {min(0.80, MAX_UTIL):.2f}'i asamaz: "
                             f"GPU 2 toplami {GPU2_PLAN_CAP_GB} GB'i gecmemeli (servis alarmi 20 GB)")
        return v

    @model_validator(mode="after")
    def models_fit(self) -> Phase3:
        if self.max_num_seqs < max(self.concurrency):
            raise ValueError("max_num_seqs en yuksek esanlilik degerinden kucuk olamaz")
        budget = GPU_GB * self.gpu_memory_utilization
        for m in self.models:
            need = m.weights_gb() + MIN_KV_GB
            if need > budget:
                raise ValueError(f"{m.name}/{m.quant}: ~{need:.1f} GB gerekli, "
                                 f"butce {budget:.1f} GB (GPU 2'ye sigmaz)")
        return self

    def runs(self) -> list[dict]:
        """Model x dil x esanlilik x tekrar. Model disarida: her model bir kez yuklenir."""
        return [
            {"model": m.name, "quant": m.quant, "repo": m.repo, "lang": lang,
             "concurrency": c, "repeat": r, "n_requests": max(16, 4 * c)}
            for m in self.models
            for lang, c, r in product(self.langs, self.concurrency, range(self.repeats))
        ]


def load(path: str | Path) -> Phase3:
    return Phase3.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Config'i dogrula ve plani goster (GPU'ya dokunmaz)")
    ap.add_argument("config", nargs="?", default="configs/phase3.yaml")
    cfg = load(ap.parse_args().config)
    runs = cfg.runs()
    budget = GPU_GB * cfg.gpu_memory_utilization
    print(f"{len(cfg.models)} model, {len(runs)} kosu, butce {budget:.1f} GB\n")
    for m in cfg.models:
        print(f"  {m.name:14s} {m.quant:10s} agirlik ~{m.weights_gb():5.1f} GB  "
              f"KV icin ~{budget - m.weights_gb():5.1f} GB")
    print(f"\nGPU 2 toplam: vLLM {budget:.1f} + ek yuk {OVERHEAD_GB} "
          f"+ servis payi {SERVICE_RESERVE_GB} = {budget + OVERHEAD_GB + SERVICE_RESERVE_GB:.1f}"
          f" GB  (plan tavani {GPU2_PLAN_CAP_GB}, guard 17, servis alarmi 20)")


if __name__ == "__main__":
    main()
