[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/saeeew/JP-HomophoneBench)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/saeeew/J-PACF-YOMI-tdt)


# Parakeet Contextual Fusion Research JP

Japanese contextual-ASR research suite for `nvidia/parakeet-tdt_ctc-0.6b-ja`, with a reproducible `JP-HomophoneBench` builder, Hugging Face publication pipeline, category-aware E00-E07a evaluation flow, and append-only experiment evidence in `saeeew/J-PACF-YOMI-tdt-bucket`.

## Experiment ladder

```text
E00 TDT greedy
 -> E01 TDT beam
 -> E02 + KenLM-derived NGPU-LM
 -> E03 + GPU-PB / TurboBias-style context biasing
 -> E04 + local hybrid-CTC N-best rerank
 -> E05 + frozen-encoder phoneme CTC rerank
 -> E06 optional in-beam scorer integration
 -> E07a Shisa V2 deterministic N-best selection
```

The Parakeet checkpoint remains frozen through E04. E05 trains only a small phoneme CTC projection over cached FastConformer encoder states. E07a does not modify the ASR model or generate replacement text: `shisa-ai/shisa-v2-qwen2.5-7b` selects exactly one hypothesis that already exists in the upstream N-best list.

See [`docs/e07a-shisa-nbest-selector.md`](docs/e07a-shisa-nbest-selector.md) for the leakage controls, canonical run procedure, metrics, decision rules, and HF Bucket evidence workflow.

## JP-HomophoneBench core8

The benchmark separates eight error/disambiguation classes instead of collapsing them into CER:

- `exact_homophone`
- `near_homophone`
- `voicing`
- `long_vowel`
- `geminate`
- `moraic_nasal`
- `pitch_accent`
- `semantic_only`

Every row stores phonetic/contextual difficulty metadata and source provenance. Exact homophones intentionally have phone distance `0` while remaining acoustically unresolvable at the segmental-phone level.

## Published Hugging Face configs

Publication target: `saeeew/JP-HomophoneBench`.

### `homophone8`

The default public/permissive config excludes NonCommercial-derived rows. Open-license pitch-accent rows are combined with a small CC0 synthetic metadata supplement so all core8 classes remain represented.

```text
records:    113
train:       97
validation:   7
test:         9
core8: complete
```

### `homophone8-research`

The larger research config includes CC BY-NC 4.0-derived rows from `IDEMITSU/hoiku-yougo-stt-ja` and must be used according to its upstream license.

```text
records:    660
train:      439
validation: 69
test:       152
core8: complete
```

See `docs/releases/v0.1.0.md` for publication provenance, source counts, Actions run IDs, and artifact SHA-256.

## Setup

The recommended local development path is the repository Dev Container. Open the repository root in VS Code/Cursor and run **Dev Containers: Reopen in Container**. The container restores Python 3.12.3 and uv 0.12.1 from `mise.lock`, materializes the locked CPU/static environment, and prepares the isolated Hugging Face Bucket CLI environment.

```bash
mise run ci
```

See [`.devcontainer/README.md`](.devcontainer/README.md) for cache/state volumes, Hugging Face authentication, ARM-host notes, and the boundary between local CPU development and the canonical GPU runtime.

For a manual Linux/x86_64 setup, use the same repository contract rather than an ad-hoc `pip install`:

```bash
mise --locked install
mise run deps:sync
mise run hf:transport:sync
mise run test
```

GPU experiments additionally require the authoritative NVIDIA NeMo/CUDA runtime. The default Dev Container intentionally does not duplicate that environment. E07a keeps the repository's authoritative `uv.lock` unchanged and layers versioned Transformers packages onto the locked GPU runtime with `uv run --with`.

## Build and publish the fixed Dataset

```bash
make bench-permissive
make bench-validate
```

Manual Hugging Face publication is available through `.github/workflows/jp-homophone-hf-publish.yml`. The repository secret `HF_TOKEN` must have Dataset write permission.

Local publication:

```bash
export HF_TOKEN=hf_...
make hf-publish HF_CONFIG=homophone8 HF_LICENSE_POLICY=permissive
```

For the mixed-license research config:

```bash
make hf-publish HF_CONFIG=homophone8-research HF_LICENSE_POLICY=research
```

## Evaluation data flow

The Hub Dataset is metadata-first: upstream audio is not copied into `saeeew/JP-HomophoneBench`. Evaluation therefore has two distinct stages.

```text
saeeew/JP-HomophoneBench
        |
        | load_dataset(config, split)
        v
bench_index.jsonl
  |          |          |
  |          |          +--> lm_corpus.txt
  |          +-------------> context_phrases.txt
  |
  +--> audio_ref -> upstream HF Dataset -> local audio/
                                      |
                                      v
                               nemo_eval.jsonl
```

Fetch only the published benchmark index, without downloading source audio:

```bash
make hf-eval-index HF_CONFIG=homophone8 HF_SPLIT=test
```

Rehydrate source audio for runnable rows and build the NeMo manifest:

```bash
make hf-eval-audio HF_CONFIG=homophone8 HF_SPLIT=test
```

Generated files:

```text
data/generated/
├── bench_index.jsonl
├── nemo_eval.jsonl
├── context_phrases.txt
├── lm_corpus.txt
├── eval_provenance.json
└── audio/
```

Rows without a rehydratable `audio_ref` remain in `bench_index.jsonl` but are intentionally absent from `nemo_eval.jsonl`. This keeps lexical/synthetic fixtures available for taxonomy and context-list tests without pretending they are acoustic ASR samples.

## Run ASR experiments

```bash
export NEMO_ROOT=/opt/NeMo
export MODEL_NAME=nvidia/parakeet-tdt_ctc-0.6b-ja
export MANIFEST=data/generated/nemo_eval.jsonl
export CONTEXT_PHRASES=data/generated/context_phrases.txt

bash experiments/E00_tdt_greedy.sh
bash experiments/E01_tdt_beam.sh
```

For E02+ provide an NGPU-LM artifact:

```bash
export NGPU_LM=artifacts/lm/ja-6gram.nemo
bash experiments/E02_ngpulm.sh
bash experiments/E03_gpu_pb.sh
bash experiments/E04_ctc_rerank.sh
```

E05 additionally requires a trained phone head and cached encoder features. E06 remains version-isolated because it reaches into NeMo decoder internals.

The GPU-PB configuration follows current NeMo `malsd_batch` paths: `rnnt_decoding.malsd.boosting_tree.*`. NGPU-LM remains under `rnnt_decoding.beam.*`.

## Run E07a Shisa N-best selection

The canonical E07a arm consumes the E05 N-best/reranked JSONL and does not expose the benchmark reference or ASR scores to Shisa. Candidate prompt order is deterministically shuffled to reduce rank-position bias.

```bash
export INPUT=results/E05_phone_rerank.jsonl
export BENCHMARK_INDEX=data/generated/bench_index.jsonl
uv run --locked \
  --extra gpu \
  --with 'transformers==4.57.3' \
  --with 'accelerate>=1.10,<2' \
  --with 'safetensors>=0.5' \
  bash experiments/E07a_shisa_select.sh
```

Outputs:

```text
results/E07a_shisa_select.jsonl
results/E07a_metrics.parquet
results/E07a_summary.json
```

Publish immutable evidence to the project Bucket:

```bash
export HF_TOKEN=hf_...
RUN_ID="e07a-shisa7b-k8-seed7-$(git rev-parse --short=8 HEAD)"
bash scripts/hf/publish_e07a_run.sh "${RUN_ID}"
```

Remote evidence lives under:

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/runs/<RUN_ID>
```

## Collect E00-E06 metrics

The general result collector accepts NeMo single-best manifests and the repository's N-best/reranked JSONL files, joins them to the immutable benchmark IDs, and writes Zstd Parquet plus a JSON summary.

```bash
make metrics \
  RESULT_SPECS='E00=results/E00_tdt_greedy.json E01=results/E01_tdt_beam.json'
```

Outputs:

```text
results/
├── metrics.parquet
└── summary.json
```

Currently aggregated metrics include:

- full-utterance CER
- target/entity accuracy
- distractor false-positive rate
- MRR when N-best hypotheses are available
- Oracle@1/4/8/16/32
- Bias FPR when negative-control rows are present
- paired CER/entity-accuracy deltas versus E00
- the same aggregates per core8 category

E07a has a selector-specific evaluator because it returns one selected hypothesis while preserving the original N-best list as evidence. It reports source-vs-selected CER/entity accuracy, wins/losses/ties, parse/fallback rates, and category breakdowns.

This makes the main contextual-ASR diagnostic explicit:

- high Oracle@K + low Acc@1 -> scoring/ranking bottleneck;
- low Oracle@K -> search/acoustic hypothesis bottleneck;
- near-homophone gains after E05 -> phoneme scorer is helping where it should;
- exact-homophone/semantic-only remaining hard after E05 -> linguistic/entity context is the remaining bottleneck;
- E07a gains on exact-homophone/semantic-only with low damage -> second-pass semantic selection is justified.

## GitHub Actions

The repository keeps CPU checks, GPU evaluation, dataset/model publication and HF Bucket evidence flows separate. The HF Bucket is configured as `saeeew/J-PACF-YOMI-tdt-bucket`; immutable run evidence is stored below `runs/`.

Existing core workflows include:

```text
parakeet-context-fusion-ci.yml
  CPU lint / unit / compile / shell validation

devcontainer-smoke.yml
  build the local CPU/static development image
  -> verify linux/amd64 + vscode user + pinned mise prerequisites

homophone-eval-smoke.yml
  pull public HF homophone8:test
  -> build index/context/corpus
  -> identity prediction smoke
  -> metrics.parquet + summary.json

homophone-eval-gpu.yml
  manual self-hosted GPU evaluation

hf-bucket-candidate-publish.yml
  validated candidate artifact publication
```

## Repository layout

```text
.devcontainer/         local CPU/static development environment
.github/workflows/    CPU CI, HF publication, smoke and GPU evaluation
configs/              experiment/evaluation/storage defaults
data/                 benchmark metadata and CC0 seeds
docs/                 research protocols and release provenance
experiments/          E00-E07a runners
patches/              E06 NeMo integration contract
schemas/              benchmark JSON Schema
scripts/              builders, selectors, materializers, validators and metrics
src/                  reusable Python package
tests/                CPU regression tests
```

See `docs/jp-homophone-bench.md`, `docs/phone-head.md`, `docs/hf-storage.md`, `docs/e07a-shisa-nbest-selector.md`, and `docs/releases/v0.1.0.md` for deeper contracts and provenance.
