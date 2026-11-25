# Datasheet: TR-LLM Bench eval set

Format follows Gebru et al., "Datasheets for Datasets" (2021), shortened to what applies here.
Hugging Face: [HaticeB/tr-llm-bench-eval](https://huggingface.co/datasets/HaticeB/tr-llm-bench-eval).

## Motivation

The set measures whether quantization, KV-cache precision or model choice changes Turkish
accuracy differently from English accuracy. Every item exists in Turkish and English with the same
`id`, so differences are measured on paired items. Scoring is deterministic (no LLM judge).

## Composition

550 items: 250 TR/EN pairs plus 50 Turkish-only morphology items. Eight pairs are excluded from
every reported number because the Turkish translation changes the meaning (see Errata), which
leaves **242 pairs**.

| Category | Pairs (TR + EN) | Source | Type | License | Scorer |
|---|---|---|---|---|---|
| reading | 60 | [Belebele](https://huggingface.co/datasets/facebook/belebele) | public | CC-BY-SA-4.0 | choice letter |
| knowledge | 40 | [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU), culturally agnostic (CA) subset | public | Apache-2.0 | choice letter |
| math | 40 | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) / [GSM8K-TR v0.2](https://huggingface.co/datasets/malhajar/gsm8k_tr-v0.2) | public | MIT / not stated | final number |
| instruction | 60 | this project, templates in `trbench/eval/ifeval.py` | original | CC-BY-4.0 | rule checks |
| json | 50 | this project, templates in `trbench/eval/json_tasks.py` | original | CC-BY-4.0 | field match |
| morphology | 50 (TR only) | this project, `trbench/eval/morphology.py` | original | CC-BY-4.0 | exact suffix |

Selection from public sets uses a fixed seed (2026) and only items that exist in both languages.
Global-MMLU items are drawn from the CA subset so that answers do not depend on culture-specific
knowledge. GSM8K pairs are matched question by question; the 40 pairs are pinned in
`trbench/eval/gsm8k_pairs.json` with dataset revisions and per-item text hashes (TR `f05ded6`,
EN `740312a`), and the build stops if a hash does not match.

## Collection and translation

- **Belebele:** passages come from FLORES-200; the Turkish questions are part of the official
  release. The dataset card does not describe the translation workflow; the paper (Bandarkar et al.,
  2023) does.
- **Global-MMLU:** the dataset card says it combines machine translations with professional
  translations and crowd-sourced post-edits. It does not say which of these the Turkish subset
  received. *Open question for the paper.*
- **GSM8K-TR v0.2:** the card says the data was generated with GPT-4 and corrected by human
  experts. It does not state a license. Question text is therefore not redistributed here.
  *Open question: ask the author before the paper is submitted.*
- **Original items:** generated from templates in this repository, in Turkish and English in
  parallel (same topic, same constraint). Morphology covers city names with case suffixes and plural
  formation only; verb inflection, tense, negation and possessives are not covered.

## Errata

All 38 items that Qwen2.5-7B (fp16) answered correctly in English but wrongly in Turkish were
reviewed by hand: 30 are model errors, 8 are translation errors (5 Belebele, 2 Global-MMLU,
1 GSM8K-TR). They are listed in `errata.jsonl` on Hugging Face and in
`TRANSLATION_ERRORS` (`trbench/eval/analysis.py`). Only items that one model got wrong in Turkish
were reviewed, so more translation errors may remain in the set.

## Contamination

Public items (reading, knowledge, math) are on the web and may be in a model's training data.
Original items were first published with this project (November 2026), after the training
cutoff of every evaluated model. `python -m trbench.eval.contamination` reports the two kinds
separately (14 models, GPU run, 132 public and 110 original items per language):

- Model rank order agrees between the two kinds (Spearman: TR +0.64, EN +0.45).
- The mean Turkish–English gap is −18.2 points on public items and −11.3 on original items.

The two kinds measure different skills (reading and math vs. instruction following and JSON),
so the comparison is an indicator, not a contamination test.

## Distribution

Only original items and IDs of public items are published. Third-party question text is not
redistributed; `python -m trbench.eval.build` rebuilds the full 550-item set from the pinned
sources.

## Limitations

- 242 pairs: 95% confidence intervals on accuracy differences are 4–7 points wide.
- One annotator reviewed the errata; no inter-annotator agreement.
- Original items are template-based and narrow (one topic list, a fixed set of constraints).
- Turkish items on public sets are translations from English, not text written in Turkish.
