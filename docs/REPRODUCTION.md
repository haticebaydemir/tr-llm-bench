# Reproduction check (Faz 8g)

**Result:** every README table rebuilds from the committed result files in a clean environment.
No GPU second machine exists and no GPU is rented, so model runs were not repeated; this check covers
everything downstream of the raw results.

## What was run

- Fresh clone of commit `efc16e6`, new virtual environment, `pip install -r env/requirements-server.lock`,
  `pip install -e . --no-deps` (Linux, Python 3.13; the lock was written on the server with Python 3.12).
- `pytest`: 191 passed.
- `python scripts/reproduce_tables.py`: 11 of 11 table modules ran without error.

## Compared against the README

| Table | Module | Matches README |
|---|---|---|
| Tokenizer tax (§1), TR/FI/HU/DE/FR | `tokenizer_tax.langs` | yes |
| Quantization tax, 7 models (§3) | `eval.analysis --all --clean` (from `phase4_scores.jsonl`) | yes |
| 14 models on GPU (§3) | `eval.gpu_quality_report` | yes |
| KV-cache quantization (§3) | `eval.kv_report` | yes |
| Public vs original items (§3) | `eval.breakdown` | yes |
| Serving, 7 models, cache off (§2a) | `eval.gpu_report` | yes (44–100 content tok/s, 2.3×) |
| Engine comparison, caches on (§2b) | `eval.gpu_report` | yes, after a fix (below) |
| Context length, context window tax (§2d, 2d+) | `eval.gpu_report`, `eval.context_tax` | yes |
| Quality vs speed on CPU (§4) | `eval.pareto` | yes |

Figures (`docs/*.png`) are re-rendered with different bytes (matplotlib version); their content is the
same data.

## Found and fixed

`eval.gpu_report` built the engine comparison from the cache-off llama.cpp file and the cache-on vLLM
file, which overstated vLLM's lead (e.g. +108% instead of +33% for Turkish at 4 requests). The engine
table now uses the cache-on files for both engines, matching README §2b; the cache-off comparison stays
in §2b *Cold cache*. The README numbers were already correct; only the script output was wrong.

## Not covered

- Raw model runs (GPU or CPU): need the same V100 and llama.cpp `b12470`.
- Tokenizer counts from text: FLORES+ is gated and not redistributed; `tokenizer_counts.csv.gz` is the
  starting point. `python -m trbench.tokenizer_tax.fertility` rebuilds it for anyone with FLORES+ access.
- The Trendyol 1.22 value (older transformers version), see `docs/CLAIMS.md` T8.
