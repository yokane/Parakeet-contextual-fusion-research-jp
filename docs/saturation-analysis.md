# TurboBias / GPU-PB saturation analysis

This stage turns the qualitative rule "boosting is saturated when stronger bias stops helping but false positives keep rising" into a paired statistical decision over the exact same evaluation rows.

## Inputs

`analyze_saturation.py` consumes the row-level Parquet emitted by `collect_experiment_metrics.py` and a sweep specification such as `configs/saturation.example.json`.

Each sweep point identifies one experiment result and one ordered parameter value, for example:

```json
{
  "experiment": "E03_pb_1.00",
  "value": 1.0
}
```

All compared experiments should come from the same immutable `bench_index.jsonl` / `nemo_eval.jsonl` so the comparison is paired by `benchmark_id`.

## Decision rule

For adjacent settings `a -> b` the analyzer computes:

1. paired mean change in a benefit metric, normally `entity_accuracy`;
2. paired bootstrap confidence interval for that benefit change;
3. exact McNemar test over entity-correct / entity-wrong outcomes;
4. paired mean change and bootstrap CI for a risk metric, normally `distractor_false_positive_rate`;
5. the same analysis per core8 category.

The default saturation transition requires both:

```text
benefit CI upper bound <= +0.01
AND
McNemar finds no directional entity gain
AND
risk mean increase >= +0.02
AND
risk CI lower bound > 0
```

Defaults:

```text
confidence          95%
bootstrap samples   10,000
McNemar alpha       0.05
minimum useful gain 1 percentage point
minimum risk rise   2 percentage points
```

The thresholds are policy parameters rather than universal constants and are exposed on the CLI.

## Why use paired tests

The same utterance can be easy or difficult across every decoding configuration. An unpaired comparison wastes that structure. Pairing by `benchmark_id` measures whether changing only the contextual-bias setting changes the outcome for the same item.

For binary entity correctness, the informative observations are the discordant pairs:

```text
win:  old wrong -> new correct
loss: old correct -> new wrong
```

The exact McNemar test asks whether wins and losses are asymmetric. This is especially useful when the benchmark or an individual homophone category is still small.

## Category interpretation

A single global saturation point can hide the reason for failure. Read the `by_category` transitions as follows:

| Pattern | Interpretation | Next step |
|---|---|---|
| near-homophone improves, exact-homophone plateaus | acoustic/phonetic scorer may still help near matches; exact homophones need context | test E05, then linguistic/entity context |
| all phonetic classes plateau while FPR rises | lexical phrase boosting is exhausted | stop increasing `PB_ALPHA` |
| exact-homophone improves but semantic-only does not | local lexical context helps but broader discourse is missing | add document/entity prior |
| Oracle@K is high while entity Acc@1 is low | search contains the right answer but ranking is weak | improve rescoring/fusion |
| Oracle@K is low | the target is missing from the beam | increase/search-diversify beam or add acoustic/phoneme evidence |

## Running locally

After collecting a parameter sweep into `results/metrics.parquet`:

```bash
make saturation \
  RESULTS_DIR=results \
  SATURATION_SPEC=configs/saturation.example.json
```

Equivalent direct command:

```bash
python scripts/analyze_saturation.py \
  --metrics results/metrics.parquet \
  --sweep configs/saturation.example.json \
  --output results/saturation.json \
  --bootstrap-samples 10000 \
  --confidence 0.95 \
  --min-gain 0.01 \
  --min-risk-increase 0.02 \
  --alpha 0.05
```

## Self-hosted GPU sweep

`.github/workflows/homophone-saturation-gpu.yml` automates the first practical saturation experiment:

```text
fixed HF config + split
      |
      v
fixed local audio / NeMo manifest
      |
      v
E03 PB_ALPHA = 0.00
             = 0.25
             = 0.50
             = 1.00
             = 1.50
             = 2.00
      |
      v
one metrics.parquet
      |
      v
paired bootstrap + McNemar
      |
      v
saturation.json
```

The workflow is manual and requires a runner labelled:

```text
self-hosted
linux
gpu
```

It also requires the NGPU-LM file supplied by `ngpu_lm_path` and the existing NeMo environment under `NEMO_ROOT=/opt/NeMo`.

For the most statistically useful initial run, use `homophone8-research:test`, because the public permissive config intentionally contains only a small number of rows for most non-pitch categories. The mixed-license result artifact must remain subject to the upstream NonCommercial licensing constraints documented in the repository.

## Next stress axis

`PB_ALPHA` saturation answers whether stronger boosting is useful for a fixed phrase list. It does not yet answer scalability with distractors. The next independent sweep should hold `PB_ALPHA` fixed and vary context-list size, while always retaining the target/hard-homophone set and adding deterministic unrelated distractors. That distinguishes:

```text
boost-strength saturation
from
candidate-list / distractor saturation
```
