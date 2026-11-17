---
license: cc-by-4.0
language:
- tr
- en
task_categories:
- text-generation
- question-answering
pretty_name: TR-LLM Bench eval set
size_categories:
- n<1K
---

# TR-LLM Bench eval set

Parallel Turkish/English evaluation items used in
[TR-LLM Bench](https://github.com/haticebaydemir/tr-llm-bench) to measure whether
quantization hurts Turkish more than English. All scoring is deterministic (no LLM judge).

## Files

| File | Content | License |
|---|---|---|
| `items.jsonl` | 270 original items: instruction following (60 TR + 60 EN), JSON extraction (50 TR + 50 EN), Turkish morphology (50 TR) | CC-BY-4.0 |
| `references.jsonl` | IDs of the 280 third-party items (Belebele, Global-MMLU culturally-agnostic, GSM8K / GSM8K-TR) | see sources |
| `errata.jsonl` | 8 items whose Turkish translation changes the meaning, found by manual review | CC-BY-4.0 |

Third-party question text is **not** redistributed. Rebuild the full 550-item set with
`python -m trbench.eval.build` from the repository; item IDs match `references.jsonl`.

## Item format

`id` (same in TR and EN → pairs), `category`, `lang`, `prompt`, `answer`, `scorer`, `source`, `meta`.

## Translation errata

Manual review of every item that Qwen2.5-7B (fp16) answered correctly in English but
wrongly in Turkish found 8 translation errors out of 38 (5 Belebele-TR, 2 Global-MMLU-TR,
1 GSM8K-TR). Removing them roughly halves the Turkish–English reading gap. Details in
`errata.jsonl`.

## Sources

- Belebele (Meta), CC-BY-SA-4.0
- Global-MMLU (Cohere For AI), Apache-2.0
- GSM8K (OpenAI), MIT; GSM8K-TR v0.2 (malhajar), license not stated
