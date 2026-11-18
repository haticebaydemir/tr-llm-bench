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

Instruct models, GGUF built from the original weights, llama.cpp `b12470`, CPU, greedy
decoding. fp16 accuracy (%, TR / EN), with the 8 mistranslated items removed (see below):

| Model | Reading | Knowledge | Math | Instruction | JSON | Morphology (TR) |
|---|---|---|---|---|---|---|
| Gemma-2-9B | 85 / 95 | 61 / 66 | 97 / 97 | 98 / 85 | 100 / 100 | 88 |
| Qwen2.5-7B | 85 / 95 | 45 / 68 | 79 / 95 | 78 / 75 | 100 / 100 | 64 |
| Llama-3.1-8B | 67 / 91 | 58 / 61 | 79 / 90 | 67 / 97 | 100 / 100 | 66 |
| Trendyol-7B v1.8 | *re-running* | | | | | |

Quantization tax over 242 paired items (percentage points, positive = Turkish loses more,
95% bootstrap CI):

| Model | Q8_0 | Q4_K_M |
|---|---|---|
| Gemma-2-9B | +0.8 [−1.2, +2.9] | −0.4 [−3.3, +2.5] |
| Qwen2.5-7B | +1.2 [−1.7, +4.5] | +1.7 [−2.1, +5.4] |
| Llama-3.1-8B | −0.8 [−4.5, +2.9] | −1.7 [−7.4, +3.7] |

- **No measurable quantization tax in any of the three models.** Every interval includes
  zero: down to Q4_K_M, Turkish does not degrade more than English.
- **The language gap is set by the base model, and it varies a lot.** Gemma-2-9B is close
  to parity in math and knowledge and scores 88% on Turkish morphology; Qwen2.5-7B trails
  by 24 points in knowledge; Llama-3.1-8B trails by 24 in reading and by 30 in Turkish
  instruction following.
- JSON extraction shows no gap in any model.
- Trendyol-7B v1.8 (Mistral-based) ships no chat template. llama.cpp silently fell back to
  ChatML and the model answered nonsense; it is being re-run with its `[INST]` format
  (`configs/chat_templates/mistral_inst.jinja`). Check the template of every converted model.

### How much of the gap is the benchmark's fault?

All 38 items that fp16 got wrong in Turkish but right in English were reviewed by hand:
**30 are genuine model errors, 8 are translation errors** that change the meaning of
the Turkish item, and 0 are scoring errors. Examples of meaning-changing errors:

- Belebele-TR: a "not true" option was negated in translation, so it became true;
  "rotation" became "orbit" in the passage but "rotation" in the options;
  "airline" became "airport" in the options.
- Global-MMLU-TR: "EXCEPT" was dropped, reversing the question; "H2O(s)" (solid)
  became "H2O(lar)" (a Turkish plural suffix).
- GSM8K-TR: "200 taps per minute with each foot" became ambiguous ("with both feet").

Turkish–English gap at fp16 after removing those 8 items:

| Category | Raw gap | Corrected gap |
|---|---|---|
| Reading | 16.7 pts | 9.0 pts |
| Knowledge | 27.5 pts | 23.7 pts |
| Math | 17.5 pts | 15.4 pts |

**About half of the reading gap comes from Belebele-TR translation quality, not from the
model.** The knowledge and math gaps are mostly real. Several other items contain term
errors that do not change the answer (e.g. "24 hours" → "24 seconds",
"discount rate" → "indirim oranı" instead of "reeskont oranı").

## Summary

For a Turkish user of open 7–9B models: the engine serves Turkish as fast as English in
tokens, but the tokenizer delivers up to ~38% less content per second; quantization down to
Q4_K_M costs Turkish no more than English in any model tested; and the remaining quality gap
lies in the base model (Gemma-2-9B small, Llama-3.1-8B large), partly inflated by translated
benchmarks.

Data-quality notes found while inspecting failures: translationese in Belebele-TR,
meaning errors in GSM8K-TR, term errors in Global-MMLU-TR.

## Limitations

- GPU serving covers Qwen2.5-7B (fp16, AWQ-int4) and a partial Qwen2.5-14B run on one V100.
  The planned engine comparison and multi-model GPU matrix were dropped: the GPU was shared
  with a production service, and the benchmark never runs at its expense.
- Quality is measured on CPU with llama.cpp (`b12470`), greedy decoding, one run per configuration.
  GGUF files are built from the original weights with that same release.
- 250 paired items give 95% intervals of about ±4 points; a smaller quantization tax cannot be excluded.
- Reading, knowledge and math use translated benchmarks; see the errata above.

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
