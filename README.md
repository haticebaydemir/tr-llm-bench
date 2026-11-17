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

*Results table: TODO*

Finding: GPTQ-int8 does not run on V100 (SM70) for models with attention bias
(e.g. Qwen), because the Marlin kernel does not support bias.

## 3. Quantization tax on Turkish quality (llama.cpp, CPU)

550 items: parallel TR/EN reading (Belebele), knowledge (Global-MMLU,
culturally agnostic subset), math (GSM8K / GSM8K-TR), instruction following,
JSON extraction, plus a Turkish-only morphology set.

**Quantization tax = (accuracy loss in TR) − (accuracy loss in EN)** relative to fp16,
with bootstrap 95% confidence intervals.

*Results table (fp16 / Q8_0 / Q4_K_M): TODO*

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
