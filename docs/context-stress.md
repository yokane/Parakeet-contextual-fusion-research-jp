# Acoustic coverage and context-list stress protocol

This document defines the next experiment after the E03 phrase-boosting strength sweep.

## 1. Do not confuse benchmark rows with runnable acoustic rows

`JP-HomophoneBench` intentionally contains several kinds of sources:

- real audio rows;
- dictionary/lexical rows;
- synthetic error pairs;
- manually curated homophone/context fixtures.

Only rows that can be rehydrated to an actual audio file belong in an acoustic ASR experiment. `scripts/materialize_hf_eval.py` already writes both the complete benchmark index and a smaller `nemo_eval.jsonl` containing runnable audio rows.

The generated `eval_provenance.json` records:

```json
{
  "records": 660,
  "runnable_audio_records": 0,
  "categories": {},
  "runnable_categories": {}
}
```

The numbers above are illustrative only. Always use the generated provenance for the selected Hub revision/config/split.

Before a GPU saturation or context stress run, `validate_audio_coverage.py` now checks the real runnable counts.

Example:

```bash
python scripts/validate_audio_coverage.py \
  --provenance data/generated/eval_provenance.json \
  --required-category exact_homophone \
  --required-category near_homophone \
  --min-per-category 5 \
  --min-total 10 \
  --output results/audio_coverage.json
```

If a required category has too little real audio, the workflow fails **before inference**. A large metadata benchmark must never be reported as a large acoustic benchmark when only a small subset was decoded.

## 2. Separate phrase-boost strength from context-list size

There are two different stress axes:

### A. Phrase-boost strength

```text
PB_ALPHA = 0.0, 0.25, 0.5, 1.0, 1.5, 2.0
```

Hold context phrases fixed. This answers:

> At what boost strength does entity benefit plateau while false positives start increasing?

Workflow:

```text
homophone-saturation-gpu
```

The sweep is normalized to sorted unique values and must include `0.0` as the no-boost baseline.

### B. Additional unrelated distractors

Hold `PB_ALPHA`, audio, target phrases, hard negatives, LM, beam parameters, and model fixed. Change only the number of unrelated context phrases.

Recommended initial axis:

```text
D = 0, 10, 100
```

After a larger external lexicon is available:

```text
D = 0, 10, 100, 1000, 10000
```

`D` means **additional unrelated distractors**, not total context-list size.

This distinction matters because every experiment must retain the target and its hard homophone/near-homophone candidates. A nominal `context_size=1` experiment would otherwise remove the hard negatives and change the task itself.

## 3. Deterministic nested context lists

Generate stress lists with:

```bash
python scripts/build_context_stress.py \
  --benchmark data/generated/context-stress/bench_index.jsonl \
  --execution-manifest data/generated/context-stress/nemo_eval.jsonl \
  --output-dir data/generated/context-stress/lists \
  --distractor-counts 0,10,100 \
  --seed 20260901
```

The builder creates:

```text
lists/
├── context_d00000.txt
├── context_d00010.txt
├── context_d00100.txt
└── context_stress_manifest.json
```

For all cases:

```text
required = every target/candidate phrase used by runnable rows
```

The distractor pool comes from non-runnable benchmark phrases plus an optional external newline-delimited lexicon. Required phrases are removed from the distractor pool.

The pool is ordered deterministically using SHA-256 of:

```text
seed || phrase
```

and each larger case takes a longer prefix. Therefore:

```text
D0 ⊂ D10 ⊂ D100 ⊂ D1000
```

with the same required phrase set in every condition.

If the requested pool is larger than available phrases, the builder fails instead of silently reporting a smaller experiment. Use `--external-distractors` to provide a larger lexicon. `--allow-short-pool` exists only for exploratory use and should not be used for a published benchmark claim.

## 4. Context robustness envelope

The workflow aggregates each E03 run into one Parquet file and compares every distractor condition against the smallest condition with paired bootstrap confidence intervals.

Default tolerances are:

```text
entity accuracy drop <= 0.02
context distractor FPR increase <= 0.02
CER increase <= 0.01
```

The analyzer reports:

```text
results/context-stress/context_stress.json
```

including:

- overall paired deltas;
- category-specific paired deltas;
- confidence intervals;
- the maximum tested distractor count still inside the configured robustness envelope.

The result should be interpreted as an empirical operating range, not as a universal model limit.

## 5. Manual GPU workflow

Run:

```text
homophone-context-stress-gpu
```

on a runner labeled:

```text
self-hosted
linux
gpu
```

Default scientific gate:

```text
required_audio_categories = exact_homophone,near_homophone
min_audio_per_category    = 5
```

At the current dataset stage this gate may intentionally fail. That failure is useful: it means the next task is to obtain or construct appropriately licensed real/synthetic acoustic fixtures for those categories rather than pretending text-only rows were decoded.

## 6. Recommended progression

```text
Audio coverage gate
       |
       v
PB_ALPHA sweep
       |
       v
choose stable PB_ALPHA
       |
       v
extra-distractor sweep
       |
       +--> robust --> E04 local CTC / E05 phoneme scorer
       |
       +--> degrades --> candidate filtering / per-stream lists / learned context scorer
```

Do not tune PB strength and distractor count in one joint grid initially. Sequential tuning makes the failure source interpretable and avoids unnecessary GPU experiments.
