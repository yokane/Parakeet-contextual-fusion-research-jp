[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/saeeew/JP-HomophoneBench)

# Parakeet Contextual Fusion Research JP

Japanese contextual-ASR research suite for `nvidia/parakeet-tdt_ctc-0.6b-ja`, with a reproducible `JP-HomophoneBench` builder, Hugging Face publication pipeline, and category-aware E00-E06 evaluation flow.

## Experiment ladder

```text
E00 TDT greedy
 -> E01 TDT beam
 -> E02 + KenLM-derived NGPU-LM
 -> E03 + GPU-PB / TurboBias-style context biasing
 -> E04 + local hybrid-CTC N-best rerank
 -> E05 + frozen-encoder phoneme CTC rerank
 -> E06 optional in-beam scorer integration
```

The Parakeet checkpoint remains frozen through E04. E05 trains only a small phoneme CTC projection over cached FastConformer encoder states.

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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,g2p]'
pytest
```

GPU experiments additionally require NVIDIA NeMo and CUDA.

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

## Collect E00-E06 metrics

The result collector accepts NeMo single-best manifests and the repository's N-best/reranked JSONL files, joins them to the immutable benchmark IDs, and writes Zstd Parquet plus a JSON summary.

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

This makes the main contextual-ASR diagnostic explicit:

- high Oracle@K + low Acc@1 -> scoring/ranking bottleneck;
- low Oracle@K -> search/acoustic hypothesis bottleneck;
- near-homophone gains after E05 -> phoneme scorer is helping where it should;
- exact-homophone/semantic-only remaining hard after E05 -> add linguistic/document/entity context rather than increasing phoneme weight.

## GitHub Actions

Three workflows have distinct responsibilities:

```text
parakeet-context-fusion-ci.yml
  CPU lint / unit / compile / shell validation

homophone-eval-smoke.yml
  pull public HF homophone8:test
  -> build index/context/corpus
  -> identity prediction smoke
  -> metrics.parquet + summary.json

homophone-eval-gpu.yml
  manual only, runs-on [self-hosted, linux, gpu]
  -> rehydrate source audio
  -> selected E00..E06
  -> category-aware Parquet metrics
```

The GPU workflow defaults to `E00,E01`. E02-E04 need the NGPU-LM file available on the runner; E05/E06 additionally require their experiment-specific artifacts.

## Repository layout

```text
.github/workflows/    CPU CI, HF publication, smoke and GPU evaluation
configs/              experiment/evaluation defaults
data/                 benchmark metadata and CC0 seeds
experiments/          E00-E06 runners
patches/              E06 NeMo integration contract
schemas/              benchmark JSON Schema
scripts/              builders, materializers, validators, decoders and metrics
src/                  reusable Python package
tests/                CPU regression tests
```

See `docs/jp-homophone-bench.md`, `docs/phone-head.md`, and `docs/releases/v0.1.0.md` for deeper contracts and provenance.
