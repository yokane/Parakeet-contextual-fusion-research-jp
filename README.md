# Parakeet Contextual Fusion Research JP

Japanese contextual-ASR research suite for `nvidia/parakeet-tdt_ctc-0.6b-ja`, with a reproducible `JP-HomophoneBench` builder and Hugging Face publication pipeline targeting `saeeew/JP-HomophoneBench`.

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

`general` and `acoustic_stress` are auxiliary controls.

Every row stores `difficulty.acoustic`, `difficulty.lexical`, `difficulty.context`, and `difficulty.phone_distance`. Exact homophones intentionally have phone distance `0` while remaining acoustically unresolvable at the segmental-phone level.

## Published release

`saeeew/JP-HomophoneBench` currently contains the public `homophone8-research` config built from release `v0.1.0`.

```text
records:    660
train:      439
validation: 69
test:       152
core8:      complete
```

This config deliberately includes CC BY-NC 4.0-derived rows from `IDEMITSU/hoiku-yougo-stt-ja`, so it is a research/mixed-license configuration. Inspect each row's `source.license` before reuse. A complete `permissive` core8 config is not published yet because the current non-NC sources alone do not cover all eight categories.

See `docs/release-v0.1.0.md` for exact category counts, sources, and SHA-256 values.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,g2p]'
pytest
```

## Build fixed JSONL

```bash
python scripts/build_jp_homophone_bench.py \
  --output-dir data/releases/v0.1.0 \
  --semantic-tsv data/seed/semantic_homophones.example.tsv \
  --near-threshold 1.0 \
  --require-core8

python scripts/validate_jp_homophone_release.py \
  --release-dir data/releases/v0.1.0 \
  --schema schemas/benchmark.schema.json \
  --require-core8
```

Output:

```text
data/releases/v0.1.0/
├── all.jsonl
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── stats.json
```

The split is homophone-group aware: a stable SHA-256 hash maps a whole phonetic group to one split, preventing reading-family leakage between train and test.

## Upstream sources

The builder supports:

- `NagaYu/mondegreen-asr-errors`: synthetic phonetic-error taxonomy, CC0-1.0.
- `IDEMITSU/hoiku-yougo-stt-ja`: domain vocabulary/readings/mis-conversions/context, CC BY-NC 4.0.
- `HaitongSUN/prosodic-abx` / `japanese_pitch_accent`: real pitch-accent minimal-pair metadata/audio source, CC BY 4.0.
- `data/seed/semantic_homophones.example.tsv`: redistributable exact/semantic homophone seed examples.

The published v0.1.0 release contains rows from Prosodic ABX, Hoiku vocabulary, and the manual CC0 seed. The default Hub artifact is metadata-first: upstream audio is not republished; rows store `audio_ref` provenance.

## Publish to Hugging Face

The currently published complete core8 config is:

```bash
export HF_TOKEN=hf_...
python scripts/publish_hf_dataset.py \
  --release-dir data/releases/v0.1.0 \
  --repo-id saeeew/JP-HomophoneBench \
  --config-name homophone8-research \
  --license-policy research
```

Or simply:

```bash
make hf-publish
```

`research` allows NonCommercial-derived rows to remain in the generated Dataset. The publisher preserves the upstream license in every row and the Dataset Card uses a mixed/other license declaration.

A future `permissive` config should only be published after open-license sources or owned fixtures cover all core8 categories. The publisher refuses an incomplete core8 config unless explicitly overridden.

GitHub Actions also provides a manual-only `jp-homophone-hf-publish` workflow. The repository secret `HF_TOKEN` must have Hugging Face Dataset write permission.

## Load the published Dataset

```python
from datasets import load_dataset

ds = load_dataset(
    "saeeew/JP-HomophoneBench",
    "homophone8-research",
)
```

## Run ASR experiments

GPU experiments require an NVIDIA NeMo checkout:

```bash
export NEMO_ROOT=/opt/NeMo
export MODEL_NAME=nvidia/parakeet-tdt_ctc-0.6b-ja

bash experiments/E00_tdt_greedy.sh
bash experiments/E01_tdt_beam.sh
```

For E02+ build/provide NGPU-LM, then run:

```bash
bash experiments/E02_ngpulm.sh
bash experiments/E03_gpu_pb.sh
bash experiments/E04_ctc_rerank.sh
bash experiments/E05_phone_rerank.sh
```

E06 is intentionally version-isolated because NeMo's fully batched TDT decoder internals evolve.

## Evaluation

Report at minimum:

```text
CER / NE-CER
Entity Exact Match
Acc@1 per core8 category
MRR
Oracle@4/8/16/32
Bias FPR
RTF
P50/P95 latency
VRAM
```

Diagnostic interpretation:

- High Oracle@K but low Acc@1 -> scoring/ranking bottleneck.
- Low Oracle@K -> acoustic/search hypotheses are missing the target.
- Near-homophone improves at E05 while exact-homophone does not -> phoneme scorer is behaving as intended.
- `semantic_only` remains poor after E05 -> expected; identical phones require linguistic/contextual evidence.

## Repository layout

```text
.github/workflows/    CI and manual HF publication
configs/              experiment defaults
data/                 benchmark metadata/seeds
experiments/          E00-E06 runners
patches/              E06 NeMo integration contract
schemas/              JSON Schema
scripts/              builders, validators, decoders, rerankers
src/                  reusable Python package
tests/                CPU tests
```

See `docs/jp-homophone-bench.md` and `docs/phone-head.md` for detailed benchmark and E05 contracts.
