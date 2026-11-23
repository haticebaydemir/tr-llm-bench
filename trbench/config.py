"""Deney config'i: YAML -> dogrulanmis Python nesnesi -> calistirilacak kosu listesi.

Paylasilan sunucu kurallari burada KOD olarak uygulanir: ihlal eden config hic calismaz.

GPU 2 butcesi (19 Kasim, 16 Kasim olayindan sonra yeniden hesaplandi):
  servisin olculen 24 saatlik tepe kullanimi 7.03 GB -> pay 7.5 GB
  bizim plan tavanimiz toplam 15.5 GB (guard 17 GB'ta durdurur, servisin alarmi 20 GB)
  -> bize kalan ~8 GB: agirlik + KV + engine'in kendi ek yuku
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

GPU_GB = 32.0
BYTES_PER_PARAM = {"fp16": 2.0, "gptq-int8": 1.0, "awq-int4": 0.5, "gptq-int4": 0.5, "q4_k_m": 0.6}
# Engine'in butce DISINDA kullandigi bellek. vLLM: CUDA context + CUDA graph (16 Kasim: ~4 GB);
# --enforce-eager graph'i kapatir. llama.cpp: CUDA context + hesap tamponlari.
OVERHEAD_GB = {"vllm": 4.0, "vllm-eager": 1.5, "llamacpp": 1.0}
GPU2_PLAN_CAP_GB = 15.5
SERVICE_RESERVE_GB = 7.5
OUR_BUDGET_GB = GPU2_PLAN_CAP_GB - SERVICE_RESERVE_GB          # 8.0
MIN_KV_GB = 0.5


class ModelSpec(BaseModel):
    name: str
    quant: Literal["fp16", "gptq-int8", "awq-int4", "gptq-int4", "q4_k_m"]
    params_b: float = Field(gt=0)
    repo: str                                  # HF repo ya da yerel GGUF klasoru
    weights_gb_actual: float | None = None     # biliniyorsa gercek boyut (gomme katmani dahil)
    kv_kb_per_token: int = 128                 # llama.cpp KV tahmini (fp16 KV)
    concurrency: list[int] | None = None       # modele ozel (buyuk KV'li modeller icin)
    chat_template: str | None = None           # GGUF'ta sablon yoksa (Trendyol)

    def weights_gb(self) -> float:
        return self.weights_gb_actual or self.params_b * BYTES_PER_PARAM[self.quant]


class Phase3(BaseModel):
    engine: Literal["vllm", "llamacpp"]
    port: int
    gpu_memory_utilization: float | None = None    # sadece vLLM
    enforce_eager: bool = False                    # sadece vLLM
    extra_args: list[str] = []                     # engine'e aynen gecer (or. --max-num-batched-tokens 2048)
    max_model_len: int = Field(ge=512)             # llama.cpp: slot basina baglam
    max_tokens: int = Field(ge=16)
    max_num_seqs: int = Field(ge=1)
    concurrency: list[int]
    langs: list[Literal["tr", "en"]]
    repeats: int = Field(ge=1)
    results: str = "results/phase3.jsonl"
    models: list[ModelSpec]
    # Faz 6d (yuk testi saglamlastirma); varsayilanlar eski config'leri degistirmez
    warmup: int = Field(default=0, ge=0)             # olcumden once dil basina bu kadar istek (kaydedilmez)
    min_requests: int = Field(default=16, ge=1)      # kosu basina en az istek (n = max(bu, 4 x esanlilik))
    queue_metrics: bool = False                      # vLLM /metrics: bekleyen istek, preemption
    prompt_tokens: int | None = Field(default=None, ge=16)   # TR ve EN prompt'lari ayni token sayisina kes
    # Faz 7m: llama-server istek basina prompt onbellegi; None = sunucu varsayilani (istekte alan yok)
    cache_prompt: bool | None = None

    @field_validator("port")
    @classmethod
    def port_in_our_range(cls, v: int) -> int:
        if not 18000 <= v <= 18099:
            raise ValueError("port 18000-18099 disinda (kendi port araligimiz)")
        return v

    def overhead_gb(self) -> float:
        if self.engine == "llamacpp":
            return OVERHEAD_GB["llamacpp"]
        return OVERHEAD_GB["vllm-eager" if self.enforce_eager else "vllm"]

    def max_util(self) -> float:
        return (OUR_BUDGET_GB - self.overhead_gb()) / GPU_GB

    def conc(self, m: ModelSpec) -> list[int]:
        return m.concurrency or self.concurrency

    def need_gb(self, m: ModelSpec) -> float:
        """GPU 2'de bu modelin toplam payi (servis haric)."""
        if self.engine == "vllm":
            return GPU_GB * (self.gpu_memory_utilization or 0) + self.overhead_gb()
        ctx = self.max_model_len * max(self.conc(m))
        return m.weights_gb() + ctx * m.kv_kb_per_token / 1024**2 + self.overhead_gb()

    @model_validator(mode="after")
    def fits_shared_gpu(self) -> Phase3:
        if self.max_num_seqs < max(self.concurrency):
            raise ValueError("max_num_seqs en yuksek esanlilik degerinden kucuk olamaz")
        if self.engine == "vllm":
            u = self.gpu_memory_utilization
            if u is None or not 0 < u <= self.max_util():
                raise ValueError(f"gpu_memory_utilization {self.max_util():.3f}'i asamaz: GPU 2 toplami "
                                 f"{GPU2_PLAN_CAP_GB} GB'i gecmemeli (servis alarmi 20 GB)")
            for m in self.models:
                if m.weights_gb() + MIN_KV_GB > GPU_GB * u:
                    raise ValueError(f"{m.name}/{m.quant}: ~{m.weights_gb() + MIN_KV_GB:.1f} GB gerekli, "
                                     f"butce {GPU_GB * u:.1f} GB (GPU 2'ye sigmaz)")
        for m in self.models:
            if self.need_gb(m) > OUR_BUDGET_GB:
                raise ValueError(f"{m.name}/{m.quant}: GPU 2'de ~{self.need_gb(m):.1f} GB, "
                                 f"pay {OUR_BUDGET_GB} GB (sigmaz; servis alarmi 20 GB)")
        return self

    def fingerprint(self, m: ModelSpec) -> str:
        """Sonucu degistiren her ayarin kisa ozeti. Resume anahtarina girer:
        max_tokens ya da sablon degisirse eski kosu 'bitmis' sayilmaz."""
        tpl = Path(m.chat_template).read_text(encoding="utf-8") if m.chat_template else ""
        d = {"engine": self.engine, "repo": m.repo, "quant": m.quant, "template": tpl,
             "max_model_len": self.max_model_len, "max_tokens": self.max_tokens,
             "max_num_seqs": self.max_num_seqs, "util": self.gpu_memory_utilization,
             "eager": self.enforce_eager, "extra": self.extra_args}
        if self.queue_metrics:                   # /metrics sorgusu olcumu etkileyebilir; kapaliyken eski fp ayni
            d["queue_metrics"] = True
        for k in ("warmup", "min_requests", "prompt_tokens", "cache_prompt"):     # yalnizca degistiyse: eski fp'ler ayni kalir
            if getattr(self, k) != type(self).model_fields[k].default:
                d[k] = getattr(self, k)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def runs(self) -> list[dict]:
        """Model x dil x esanlilik x tekrar. Model disarida: her model bir kez yuklenir."""
        return [
            {"engine": self.engine, "model": m.name, "quant": m.quant, "repo": m.repo,
             "lang": lang, "concurrency": c, "repeat": r, "n_requests": max(self.min_requests, 4 * c),
             "fp": self.fingerprint(m)}
            for m in self.models
            for lang, c, r in product(self.langs, self.conc(m), range(self.repeats))
        ]


def load(path: str | Path) -> Phase3:
    return Phase3.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Config'i dogrula ve plani goster (GPU'ya dokunmaz)")
    ap.add_argument("config", nargs="?", default="configs/gpu_llamacpp.yaml")
    cfg = load(ap.parse_args().config)
    print(f"{cfg.engine}: {len(cfg.models)} model, {len(cfg.runs())} kosu\n")
    for m in cfg.models:
        need = cfg.need_gb(m)
        print(f"  {m.name:20s} {m.quant:9s} bizim pay ~{need:4.1f} GB -> GPU 2 toplam ~"
              f"{need + SERVICE_RESERVE_GB:4.1f} GB (tavan {GPU2_PLAN_CAP_GB}, guard 17, alarm 20)")


if __name__ == "__main__":
    main()
