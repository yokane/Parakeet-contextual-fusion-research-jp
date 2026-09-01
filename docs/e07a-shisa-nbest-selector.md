# E07a — Shisa V2 N-best Selector Research Protocol

## 1. Purpose

E07a adds one deliberately narrow research arm to the existing E00–E06 ladder:

```text
E00 TDT greedy
 -> E01 TDT beam
 -> E02 + NGPU-LM
 -> E03 + GPU-PB / context biasing
 -> E04 + local CTC rerank
 -> E05 + phoneme CTC rerank
 -> E06 optional in-beam integration
 -> E07a Shisa N-best selector
```

E07a does **not** let an LLM rewrite the transcript. It asks
`shisa-ai/shisa-v2-qwen2.5-7b` to select exactly one hypothesis that already
exists in an upstream N-best list.

The research question is:

> When the correct Japanese spelling already survives in the N-best set, can a
> Japanese-capable LLM improve final semantic/homophone selection without
> damaging acoustically resolved cases?

This separates a semantic ranking bottleneck from an acoustic/search bottleneck.

## 2. Assets and roles

| Asset | Role |
| --- | --- |
| `saeeew/J-PACF-YOMI-tdt` | ASR/model artifact |
| `saeeew/JP-HomophoneBench` | immutable category-aware evaluation index |
| `shisa-ai/shisa-v2-qwen2.5-7b` | second-pass Japanese N-best selector |
| `saeeew/J-PACF-YOMI-tdt-bucket` | append-only experiment evidence |
| this GitHub repository | source, protocol, scripts, reproducibility contract |

HF Bucket page:

`https://huggingface.co/buckets/saeeew/J-PACF-YOMI-tdt-bucket`

Bucket URI used by scripts:

`hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket`

## 3. What E07a is and is not

E07a is:

- second-pass only;
- candidate-selection only;
- deterministic generation (`do_sample=False`);
- one model call per benchmark row;
- restricted to the first `K` upstream hypotheses;
- evaluated against the unchanged JP-HomophoneBench IDs;
- stored as append-only run evidence in HF Buckets.

E07a is not:

- PARCO;
- shallow fusion inside the TDT beam;
- free-form ASR error correction;
- SFT/LoRA training;
- RAG;
- document-context correction by default;
- a replacement for E03–E05.

Those are intentionally left for separate future experiments so E07a has a
clear causal interpretation.

## 4. Leakage controls

This is the most important part of the protocol.

### 4.1 Never expose the benchmark reference

`JP-HomophoneBench.text` is the reference transcript. It must never enter the
Shisa prompt.

The selector therefore rejects these context fields explicitly:

- `text`
- `reference`
- `reference_text`
- `gold`
- `gold_text`
- `target_text`

If an external context field is later used, it must be information that would
actually be available at inference time, for example a project title or a
previously approved glossary. Use `--context-field` only after documenting its
provenance.

### 4.2 Do not expose the answer through candidate order

The canonical E07a run uses `stable_shuffle` rather than ASR rank order in the
prompt. The permutation is deterministic from:

```text
seed + benchmark ID + original index + hypothesis text
```

The original N-best index is preserved in selector metadata, but prompt position
cannot be used as a trivial proxy for TDT rank.

### 4.3 Do not expose ASR scores in canonical E07a

The current E07a prompt contains candidate text only. TDT/CTC/phone scores remain
stored in the input JSONL but are not shown to Shisa. This isolates semantic
selection ability.

A future score-aware LLM selector must be a separate experiment ID.

### 4.4 No repair call after malformed output

Shisa is instructed to return only:

```json
{"selected": 2}
```

If parsing fails, E07a falls back to the original ASR top-1 and records:

```text
parse_ok = false
fallback_to_source_top1 = true
```

There is no second LLM call. This keeps inference count and failure behavior
deterministic.

## 5. Environment

The repository's `uv.lock` remains the authoritative dependency contract for the
ASR runtime. E07a intentionally does not modify that lock just to add a
second-pass research model.

Run E07a with ephemeral, versioned Transformers dependencies layered on top of
the locked GPU environment:

```bash
uv run \
  --extra gpu \
  --with 'transformers==4.57.3' \
  --with 'accelerate>=1.10,<2' \
  --with 'safetensors>=0.5' \
  bash experiments/E07a_shisa_select.sh
```

This keeps the existing NeMo/CUDA lock unchanged while making the LLM runtime
version explicit in the command and run notes.

Canonical selector model:

```text
shisa-ai/shisa-v2-qwen2.5-7b
```

The initial immutable commit prefix used by the runner is:

```text
2ba1a59
```

For a publication-grade run, resolve and record the full immutable Hugging Face
commit SHA in the experiment notes before freezing results.

## 6. Prepare the benchmark and audio

The public benchmark is metadata-first. Build the local index and rehydrate only
runnable audio rows according to the existing repository procedure:

```bash
make hf-eval-index HF_CONFIG=homophone8 HF_SPLIT=test
make hf-eval-audio HF_CONFIG=homophone8 HF_SPLIT=test
```

Expected local inputs include:

```text
data/generated/bench_index.jsonl
data/generated/nemo_eval.jsonl
data/generated/context_phrases.txt
data/generated/lm_corpus.txt
```

Keep the benchmark revision pinned by the repository lock file.

## 7. Produce the upstream N-best input

The canonical E07a input is E05:

```text
results/E05_phone_rerank.jsonl
```

This makes the principal comparison:

```text
E05 phoneme-aware ranking
        vs
E07a Shisa semantic selection over the same surviving hypotheses
```

Run E00–E05 according to the existing experiment ladder. At minimum, the E05
JSONL must retain its `candidates` list and benchmark IDs.

E06 is optional and is not required for E07a.

For an explicit ablation, E07a may be pointed at E03 or E04 with `INPUT=...`,
but that must be reported as a different input arm rather than silently mixed
with the canonical E05 run.

## 8. Run E07a

Canonical command:

```bash
uv run \
  --extra gpu \
  --with 'transformers==4.57.3' \
  --with 'accelerate>=1.10,<2' \
  --with 'safetensors>=0.5' \
  bash experiments/E07a_shisa_select.sh
```

Equivalent explicit environment:

```bash
export INPUT=results/E05_phone_rerank.jsonl
export BENCHMARK_INDEX=data/generated/bench_index.jsonl
export SHISA_MODEL=shisa-ai/shisa-v2-qwen2.5-7b
export SHISA_REVISION=2ba1a59
export SHISA_TOP_K=8
export SHISA_SEED=7
export SHISA_DTYPE=bfloat16

uv run \
  --extra gpu \
  --with 'transformers==4.57.3' \
  --with 'accelerate>=1.10,<2' \
  --with 'safetensors>=0.5' \
  bash experiments/E07a_shisa_select.sh
```

Do not set `SHISA_CONTEXT_FIELD` for the canonical zero-external-context run.

Outputs:

```text
results/E07a_shisa_select.jsonl
results/E07a_metrics.parquet
results/E07a_summary.json
```

## 9. E07a JSONL contract

The original row and N-best candidates are preserved. E07a adds:

```json
{
  "selector_selected_text": "今年の気候は大きく変化した",
  "selector": {
    "experiment": "E07a",
    "model": "shisa-ai/shisa-v2-qwen2.5-7b",
    "revision": "2ba1a59",
    "top_k": 8,
    "seed": 7,
    "candidate_order": "stable_shuffle",
    "context_field": null,
    "prompt_sha256": "...",
    "selected_prompt_index": 2,
    "selected_original_index": 1,
    "source_top1_text": "今年の機構は大きく変化した",
    "changed_from_source_top1": true,
    "parse_ok": true,
    "fallback_to_source_top1": false,
    "parse_error": null,
    "raw_output": "{\"selected\": 2}",
    "generation": {
      "do_sample": false,
      "max_new_tokens": 24
    }
  }
}
```

`raw_output` is evidence, not the prediction itself. The prediction is always an
existing N-best hypothesis selected through `selected_original_index`.

## 10. Metrics

`evaluate_shisa_selector.py` compares the original upstream top-1 with the Shisa
selection and writes both per-row and aggregate evidence.

Primary fields:

- source CER;
- selected CER;
- source entity accuracy;
- selected entity accuracy;
- changed rate;
- parse failure rate;
- fallback rate;
- entity wins;
- entity losses;
- entity ties;
- CER improved/damaged/unchanged counts.

The most important safety metric is the loss/damage side, not only wins.

Conceptually:

```text
win:   E05 wrong -> E07a right
loss:  E05 right -> E07a wrong
tie:   entity correctness unchanged
```

## 11. Category-level interpretation

E07a is expected to be most informative on:

- `exact_homophone`
- `semantic_only`

These are the cases where segmental phoneme information cannot, by itself,
select the correct Japanese spelling.

The following categories are guard/control groups:

- `near_homophone`
- `voicing`
- `long_vowel`
- `geminate`
- `moraic_nasal`
- `pitch_accent`

If E07a improves semantic categories while damaging these categories, a future
selector should be gated rather than applied globally.

## 12. Use Oracle@K correctly

E07a can only select a correct answer that already survived upstream search.
Compare E05 Oracle@K with E07a selected accuracy.

Interpretation:

```text
Oracle@8 high, E07a low
  -> semantic selector/prompt/model limitation

Oracle@8 high, E07a much higher than E05 top-1
  -> ranking/semantic bottleneck; E07a is useful

Oracle@8 low, E07a unchanged
  -> correct hypothesis is absent; improve TDT/PB/CTC/phoneme/PARCO side
```

Do not interpret an E07a failure as evidence that an LLM is generally useless if
the correct candidate was never present in the input set.

## 13. Recommended decision rule

Before looking at final test results, record the primary endpoint and guard
metrics.

Recommended primary endpoint:

```text
entity accuracy delta on exact_homophone + semantic_only
```

Recommended guards:

```text
entity losses on originally correct rows
CER damage count
parse/fallback rate
non-semantic category deltas
latency and VRAM
```

A positive E07a result should show semantic-category gains without a comparable
increase in losses on already-correct rows.

For formal claims, use paired significance testing on the same benchmark IDs
(e.g. McNemar for paired correctness and paired bootstrap for CER deltas).

## 14. Publish immutable evidence to HF Buckets

HF Buckets are the experiment evidence store. They are mutable object storage,
so this repository enforces append-only run IDs under `runs/`.

Choose a unique run ID, for example:

```bash
RUN_ID="e07a-shisa7b-k8-seed7-$(git rev-parse --short=8 HEAD)"
```

Publish the completed run:

```bash
export HF_TOKEN=hf_...
bash scripts/hf/publish_e07a_run.sh "${RUN_ID}"
```

The helper stages:

```text
E07a_shisa_select.jsonl
E07a_metrics.parquet -> metrics.parquet
E07a_summary.json    -> summary.json
benchmark index
execution manifest (when present)
run-context.json
samples.jsonl
```

and then delegates to the existing validated append-only Bucket uploader.

Remote path:

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/runs/<RUN_ID>
```

The existing uploader refuses to overwrite an already existing run ID.

Current Hugging Face Bucket tooling supports directory synchronization in both
directions. To retrieve a frozen run later:

```bash
hf buckets sync \
  hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/runs/${RUN_ID} \
  ./reproduced/${RUN_ID}
```

## 15. Research work sequence

Use this order so causal interpretation remains intact:

1. Freeze Git commit and HF revisions.
2. Freeze JP-HomophoneBench config/split.
3. Materialize the same executable benchmark rows for all experiments.
4. Run E00–E05 and save N-best/Oracle evidence.
5. Run canonical E07a with no external context and stable shuffled prompt order.
6. Inspect parse/fallback failures before quality metrics.
7. Compare E05 vs E07a overall and by core8 category.
8. Verify that all E07a selections existed in the original N-best list.
9. Record latency, GPU model, VRAM, Transformers version, Shisa revision and git SHA.
10. Publish the run bundle to the HF Bucket with a new immutable run ID.
11. Only after the primary E07a result is frozen, decide whether to proceed to PARCO-like contextualization, gating, RAG, or constrained correction.

## 16. What to do after E07a

Do not add another LLM stage merely because E07a works.

Use the evidence:

- If E05 Oracle@K is high and E07a substantially improves
  `exact_homophone/semantic_only`, semantic reranking is justified.
- If Oracle@K is low, focus on candidate generation: PB, CTC, phoneme scoring or
  PARCO-like contextual ASR.
- If E07a wins and losses are both high, investigate a confidence/homophone gate.
- If parse failures are non-trivial, fix the selector protocol before changing
  the ASR model.
- If semantic gains require external context, create a separate experiment ID;
  do not retroactively redefine E07a.

E07a therefore remains a small, reproducible test of one question: whether a
Japanese LLM can pick the right existing hypothesis when the acoustic/search
system has already kept it alive.
