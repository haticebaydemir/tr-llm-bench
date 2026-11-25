# Claim–evidence matrix

Every claim the README, blog and paper make, with the experiment behind it, the result file, the
command that rebuilds the number, the commit that holds the data, and the limit of the claim.
A row without a result file in the repository is a claim the repository cannot yet support.
Status as of commit `795e141` (26 November 2026).

**Status:** *observed* = measured in this setup · *interpreted* = consistent with the data, not
tested directly · *gap* = evidence missing or weaker than the wording.

## Tokenizer

| # | Claim | Evidence (file · command · commit) | Limit | Status |
|---|---|---|---|---|
| T1 | The same content costs 0.64× to 2.21× the English tokens in Turkish, depending on the tokenizer | `tokenizer_tax.csv`, `tokenizer_langs.csv` · `trbench.tokenizer_tax.langs` · 93acff8 | FLORES+ (2,009 sentences, news/travel/wiki style); 17 tokenizers, 13 independent | observed |
| T2 | 13 of 17 tokenizers are independent; Llama-3.1/Cosmos-TR, Qwen2.5/Qwen3, Phi-4-mini/gpt-oss, Ministral/Mistral-Nemo give identical counts | `tokenizer_counts.csv.gz` · `trbench.tokenizer_tax.distribution` · 0a570fd | — | observed |
| T3 | In 11 of 13 tokenizers Turkish is cheaper than Finnish and Hungarian (TR / mean(FI, HU) 0.40–0.93); only Mistral-7B (1.10) and DeepSeek-V3 (1.06) are above 1 | same as T2 | Three agglutinative languages only | observed |
| T4 | Vocabulary size does not explain the tax (Spearman −0.28, n = 13) | same as T2 | n = 13; no significance test | observed |
| T5 | The cost depends on how much Turkish the tokenizer saw in training | T3 + T4; Kumru (Turkish-focused) 0.64 | Training mixtures are not published for most tokenizers | interpreted |
| T6 | The tax falls slightly with sentence length (Mistral-7B 2.35 → 2.08) | same as T2 | Cause (names, numbers in short sentences) not tested | observed, cause interpreted |
| T7 | The result does not depend on one part of the corpus (dev vs devtest differ by ≤ 0.018) | same as T2 | Same corpus, same domains | observed |
| T8 | The same tokenizer file gives 1.10 / 1.12 / 1.22 for Trendyol depending on the library version | README §1 note ¹ | Only Trendyol checked | observed — **gap:** no result file or script for the three values |

## Serving (V100, ~8 GB budget)

| # | Claim | Evidence | Limit | Status |
|---|---|---|---|---|
| S1 | Raw speed is similar across 7 models (95–109 tok/s at 8 requests); Turkish content speed is 44–100 content tok/s with the prompt cache off | `gpu_llamacpp_cold.jsonl` · `trbench.eval.gpu_report` · 2434155 | llama.cpp Q4_K_M, shared V100, c = 1/4/8, 3 repeats | observed |
| S2 | Cost per Turkish word differs 2.3× between models | same as S1 | Derived from S1 and T1 | observed |
| S3 | Prompt caching inflated earlier TTFT and llama.cpp throughput (+15–35% tok/s) | `gpu_llamacpp_cache.jsonl` · b2cb2b5; `gpu_6d_mnbt.jsonl` · b7d3cd9 | Mistral-7B only | observed |
| S4 | With the cache off, Turkish TTFT is 1.4–1.7× English on vLLM and 1.3–1.5× on llama.cpp (Mistral-7B, 4–8 requests) | `gpu_6d_mnbt.jsonl`, `gpu_llamacpp_cold.jsonl` · `trbench.eval.queue_report` | One model, 8 GB budget, closed-loop load | observed |
| S5 | At equal token counts (300) the TR/EN TTFT difference disappears | `gpu_6d_mnbt.jsonl` · b7d3cd9 | One model, one length | observed |
| S6 | Neither queueing, preemption nor the per-step prompt budget (2048) explains S4 | `gpu_6d.jsonl` (4f35790), `gpu_6d_mnbt.jsonl` (b7d3cd9) | Mechanism of S4 is therefore token count alone; not tested beyond 8 requests | observed |
| S7 | With caches on, vLLM serves Turkish 33–41% faster than llama.cpp at 4 requests; at 8 requests Turkish loses that advantage | `gpu_vllm_upstream.jsonl` · d782e6e | Caches on (README 2b first table); different 4-bit formats (AWQ/GPTQ vs Q4_K_M), so not a pure engine comparison; eager mode | observed, scope narrow |
| S8 | An 8,192-token window holds 3,710–7,475 English tokens' worth of Turkish; prefill up to 2.5× longer; decode 0–7% slower | `gpu_context.jsonl` · `trbench.eval.context_tax` · f79d584 | llama-bench, Q4_K_M, 3 repeats; Gemma-2-9B only at 1,024 | observed |

## Quality

| # | Claim | Evidence | Limit | Status |
|---|---|---|---|---|
| Q1 | No Turkish-specific loss from Q8_0 / Q4_K_M weights in 7 models; losses above 3.3 (Q8_0) / 5.4 (Q4_K_M) points are excluded | `results/phase4.jsonl` · `trbench.eval.stats` · ebee503 | 242 pairs; CPU llama.cpp b12470 | observed — **gap:** `phase4.jsonl` is git-ignored and not in the repository; the main quantization claim cannot be rebuilt from the repo |
| Q2 | Q8_0 KV cache: no change (loss ≤ 2.9 points excluded); q4_0: no consistent loss, Llama-3.1-8B the exception (+5.8 [+0.4, +11.2]) | `gpu_kvcache.jsonl` · `trbench.eval.kv_report`, `trbench.eval.stats` · dcee413 | 5 models; equivalence only defensible for q8_0 (TOST margin ±1.7–5.0) | observed |
| Q3 | Turkish accuracy ranges 48–90% across 14 models and does not follow size | `gpu_quality.jsonl` · `trbench.eval.gpu_quality_report` · afde675 | Q4_K_M, greedy, one run | observed |
| Q4 | Models fall into 3 tiers in Turkish (Holm-corrected McNemar); in English 9 models share the top tier | same as Q3 · `trbench.eval.stats` · ebee503 | Tiers overlap by construction (greedy from the top) | observed |
| Q5 | Gemma-2-9B leads in Turkish | same as Q3 | **Not significant** against Qwen3-8B (89.7% vs 81.4%, Holm) | fixed — README now says Gemma-2-9B and Qwen3-8B share the top tier |
| Q6 | The TR gap is not only a benchmark artefact: 9 of 14 models also trail on original items | `gpu_quality.jsonl` · `trbench.eval.breakdown`, `trbench.eval.contamination` · 795e141 | Public and original items test different skills | observed |
| Q7 | The gap is larger on public benchmarks, partly from translation errors and possibly contamination | same as Q6; 8 confirmed errors | Cannot separate translation, contamination and difficulty | interpreted |
| Q8 | 8 of 38 TR-wrong / EN-right items are translation errors | `TRANSLATION_ERRORS` in `analysis.py`; HF `errata.jsonl` | Only Qwen2.5-7B fp16 failures reviewed; one annotator | observed |
| Q9 | GPU and CPU give the same right/wrong result on 94–99% of items | `gpu_quality.jsonl` + `phase4.jsonl` · `trbench.eval.gpu_quality_report` | **gap:** needs `phase4.jsonl` (see Q1) | observed |
| Q10 | Pareto front on CPU: Gemma-2-9B, Aya-Expanse-8B, Trendyol-7B Q4_K_M | `trbench.eval.pareto` | **gap:** needs `phase4.jsonl` and CPU speed results, neither in the repository | observed |
| Q11 | Q4_K_M costs at most 1 point of Turkish accuracy in every model (README §4) | `phase4.jsonl` · `trbench.eval.stats` | Point estimate only; the excluded loss is up to 5.4 points | fixed — README §4 reworded with Q1's bound |

## Open items found by this matrix

1. **`results/phase4.jsonl` is not in the repository** (git-ignored), but Q1, Q9 and Q10 rest on it.
   Add it to `.gitignore`'s allow-list (11 MB; or gzip) — it holds model answers to public items, so check
   that answers do not reproduce third-party question text before publishing.
2. T8 (tokenizer version 1.10 / 1.12 / 1.22) has no result file or script.
3. Q5, Q11: README fixed; the leaderboard and blog still need the tier wording.
