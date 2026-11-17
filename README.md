# TR-LLM Bench

How much does Turkish cost to serve on open LLMs, and how much quality does
quantization take away from Turkish compared with English?

Everything here is measured on parallel TR/EN data with deterministic scoring
(seed 2026, no LLM judge), so the Turkish and English numbers are directly comparable.

## 1. Tokenizer tax

The same FLORES+ sentences take a different number of tokens per language.
**Tax = total Turkish tokens / total English tokens.** Fertility (tokens per word)
is not used because word lengths differ across languages.

| Model | TR/EN tax |
|---|---|
| Aya-Expanse-8B | 1.21 |
| Trendyol-7B | 1.22 |
| Gemma-3-12B | 1.36 |
| Gemma-2-9B | 1.40 |
| Llama-3.1-8B / Cosmos-TR-Llama-8B | 1.40 |
| gpt-oss-20b | 1.42 |
| Mistral-Nemo-12B | 1.48 |
| Qwen2.5-7B / Qwen3-8B | 1.62 |
| Phi-4 | 1.92 |
| DeepSeek | 2.03 |
| Mistral-7B-v0.3 | 2.21 |

A 1.62 tax means the same Turkish content costs 62% more tokens: more latency,
less effective context and lower content throughput. Trendyol's Turkish vocabulary
extension (32,768 → 44,312 tokens) cuts the tax to roughly Aya's level.

![Tokenizer tax](docs/tokenizer_tax_en.png)

## 2. Serving performance (vLLM, V100)

Load tests use fixed output length (`min_tokens` + `ignore_eos`) and streaming TTFT.
Raw tok/s is reported together with **content throughput = tok/s ÷ tax**,
because Turkish tokens carry less content than English ones.

Qwen2.5-7B-Instruct, vLLM, one V100 32 GB, 256 output tokens per request, median of 3 repeats.
**Content tok/s = TR tok/s ÷ 1.62** (Qwen's tokenizer tax), i.e. how much English-equivalent content Turkish users get.

| Quant | Concurrency | EN tok/s | TR tok/s | TR content tok/s | TTFT p50 EN / TR (ms) |
|---|---|---|---|---|---|
| fp16 | 1 | 15.5 | 15.6 | 9.6 | 84 / 84 |
| fp16 | 4 | 61.0 | 60.6 | 37.4 | 217 / 196 |
| fp16 | 16 | 185.5 | 185.5 | 114.5 | 412 / 404 |
| fp16 | 32 | 273.8 | 265.0 | 163.6 | 540 / 602 |
| AWQ-int4 | 1 | 32.6 | 31.6 | 19.5 | 57 / 60 |
| AWQ-int4 | 4 | 124.6 | 122.7 | 75.7 | 117 / 125 |
| AWQ-int4 | 16 | 352.4 | 315.5 | 194.7 | 261 / 252 |
| AWQ-int4 | 32 | 449.9 | 431.0 | 266.0 | 529 / 554 |

- **The engine is language-neutral, the tokenizer is not.** Raw tok/s is almost identical
  for Turkish and English, yet Turkish users get ~38% less content per second.
- **AWQ-int4 roughly doubles throughput** over fp16 on V100 at every concurrency level
  (single-stream decoding is memory-bound, so 4× smaller weights pay off).
- Qwen2.5-14B AWQ-int4 (Turkish only, partial run): 17.7 tok/s at c=1, 203 tok/s at c=32.

Finding: GPTQ-int8 does not run on V100 (SM70) for models with attention bias
(e.g. Qwen), because the Marlin kernel does not support bias.

## 3. Quantization tax on Turkish quality (llama.cpp, CPU)

550 items: parallel TR/EN reading (Belebele), knowledge (Global-MMLU,
culturally agnostic subset), math (GSM8K / GSM8K-TR), instruction following,
JSON extraction, plus a Turkish-only morphology set.

**Quantization tax = (accuracy loss in TR) − (accuracy loss in EN)** relative to fp16,
with bootstrap 95% confidence intervals.

Qwen2.5-7B-Instruct GGUF, llama.cpp `b12470`, CPU, greedy decoding. Accuracy (%):

| Category | n | fp16 TR | fp16 EN | Q8_0 TR | Q8_0 EN | Q4_K_M TR | Q4_K_M EN |
|---|---|---|---|---|---|---|---|
| Reading (Belebele) | 60 | 78.3 | 95.0 | 78.3 | 95.0 | 81.7 | 98.3 |
| Knowledge (Global-MMLU CA) | 40 | 42.5 | 70.0 | 42.5 | 70.0 | 45.0 | 65.0 |
| Math (GSM8K) | 40 | 77.5 | 95.0 | 80.0 | 95.0 | 72.5 | 100.0 |
| Instruction following | 60 | 78.3 | 75.0 | 78.3 | 81.7 | 78.3 | 76.7 |
| JSON extraction | 50 | 100 | 100 | 100 | 100 | 100 | 100 |
| Morphology (TR only) | 50 | 64.0 | – | 64.0 | – | 60.0 | – |

Quantization tax over 250 paired items (percentage points, positive = Turkish loses more):

| Quant | TR loss | EN loss | Tax | 95% CI |
|---|---|---|---|---|
| Q8_0 | −0.4 | −1.6 | +1.2 | [−1.6, +4.0] |
| Q4_K_M | −0.4 | −1.2 | +0.8 | [−2.8, +4.8] |

- **No measurable quantization tax for this model.** Both intervals include zero:
  down to Q4_K_M, Turkish does not degrade more than English. This holds for a 7B
  model at this sample size; a tax smaller than ~5 points cannot be ruled out.
- **The language gap is already there at fp16, and it is large:** Turkish trails English
  by 17 points in reading, 27.5 in knowledge and 17.5 in math. Quantization is not
  the problem; the base model's Turkish is.
- Instruction following and JSON extraction show no gap.
- 38 items are wrong in Turkish but right in English at fp16; these are being
  reviewed by hand to separate model errors from translation errors in the datasets.

Data-quality notes found while inspecting failures: translationese in Belebele-TR,
meaning errors in GSM8K-TR, term errors in Global-MMLU-TR.

## Reproduce

```bash
pip install -e ".[dev,tok]"
pytest -q
python -m trbench.tokenizer_tax.fertility
python -m trbench.eval.build
bash scripts/phase4_cpu.sh Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b
python -m trbench.eval.analysis --model qwen2.5-7b
```

The benchmark ran on a shared server. `trbench.guard` watches the other GPUs and a
health endpoint (`TRBENCH_SERVICE_HEALTH`) and stops the run if it disturbs anything.
