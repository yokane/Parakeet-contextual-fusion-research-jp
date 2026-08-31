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

- `NagaYu/mondegreen-asr-errors`: synthetic phonetic-error taxonomy, CC0-1.0.
- `IDEMITSU/hoiku-yougo-stt-ja`: domain vocabulary/readings/mis-conversions/context, CC BY-NC 4.0.
- `HaitongSUN/prosodic-abx` / `japanese_pitch_accent`: real pitch-accent minimal-pair metadata/audio source, CC BY 4.0.
- `data/seed/semantic_homophones.example.tsv`: redistributable exact/semantic homophone seed examples.

The default Hub artifact is metadata-first. Upstream audio is not republished; rows store `audio_ref` provenance.

## Publish to Hugging Face

```bash
export HF_TOKEN=hf_...
python scripts/publish_hf_dataset.py \
  --release-dir data/releases/v0.1.0 \
  --repo-id saeeew/JP-HomophoneBench \
  --config-name homophone8 \
  --license-policy permissive
```

`permissive` excludes NonCommercial-derived rows. For a deliberately research-only config:

```bash
python scripts/publish_hf_dataset.py \
  --release-dir data/releases/v0.1.0 \
  --repo-id saeeew/JP-HomophoneBench \
  --config-name homophone8-research \
  --license-policy research
```

GitHub Actions also provides the manual `jp-homophone-hf-publish` workflow. Add a repository secret named `HF_TOKEN` with Dataset write permission, then dispatch the workflow.

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
.github/workflows/    CI and HF publication
configs/              experiment defaults
data/                 benchmark metadata/seeds
experiments/          E00-E06 runners
schemas/              JSON Schema
scripts/              builders, validators, decoders, rerankers
src/                  reusable Python package
tests/                CPU tests
```

See `docs/jp-homophone-bench.md` and `docs/phone-head.md` for detailed benchmark and E05 contracts.
