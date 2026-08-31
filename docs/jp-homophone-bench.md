# JP-HomophoneBench: build, validate, publish

## Goal

JP-HomophoneBench separates eight Japanese ASR error/disambiguation classes so improvements can be attributed to the component that should solve them.

| class | primary signal | expected component |
|---|---|---|
| `exact_homophone` | identical phones, different surface | LM/entity context |
| `near_homophone` | small phone difference | CTC/phoneme scorer |
| `voicing` | 清音/濁音 | acoustic/phoneme scorer |
| `long_vowel` | 長音 | acoustic/phoneme scorer |
| `geminate` | 促音 | acoustic/phoneme scorer |
| `moraic_nasal` | 撥音 | acoustic/phoneme scorer |
| `pitch_accent` | lexical prosody | prosody-aware scorer |
| `semantic_only` | identical phones, context-required spelling | KenLM/long-context/PARCO-like scorer |

## Current public release

The current complete core8 release is `saeeew/JP-HomophoneBench`, config `homophone8-research`, generated from `v0.1.0`.

It contains 660 records: train 439, validation 69, test 152. Because 562 records are derived from `IDEMITSU/hoiku-yougo-stt-ja` under CC BY-NC 4.0, this is intentionally a research/mixed-license config. Every row retains `source.license` and source provenance.

A complete open/permissive core8 config is not published yet. Filtering the current release to non-NC sources would remove categories needed for core8 completeness, and the publisher is configured to refuse such an incomplete release unless explicitly overridden.

## Build

```bash
pip install -e '.[dev,g2p]'
python scripts/build_jp_homophone_bench.py \
  --output-dir data/releases/v0.1.0 \
  --semantic-tsv data/seed/semantic_homophones.example.tsv \
  --near-threshold 1.0 \
  --require-core8
```

The builder records upstream repo/config/split/source-id/revision/license, converts Japanese readings to a stable phone inventory (`Q`=促音, `N`=撥音, `:`=長音), and writes `all/train/validation/test.jsonl` plus `stats.json` with SHA-256 hashes.

## Split policy

Never split individual rows randomly. `group_id` represents the relevant phonetic family and `stable_split()` hashes the whole group with SHA-256 so one family cannot leak across train/validation/test.

## Validate

```bash
python scripts/validate_jp_homophone_release.py \
  --release-dir data/releases/v0.1.0 \
  --schema schemas/benchmark.schema.json \
  --require-core8
```

Validation checks schema, duplicate IDs, split consistency, group leakage, source-license provenance, exact-homophone phone identity, semantic context, pitch-accent annotation, and core8 completeness.

## Hugging Face publication

The release is metadata-first and does not redistribute upstream audio. Each record uses `audio_ref` to identify the original source row.

```bash
export HF_TOKEN=hf_...
python scripts/publish_hf_dataset.py \
  --release-dir data/releases/v0.1.0 \
  --repo-id saeeew/JP-HomophoneBench \
  --config-name homophone8-research \
  --license-policy research
```

The publisher uses `DatasetDict.push_to_hub()`, `embed_external_files=False`, creates a Dataset Card, and uploads `stats.json` and the JSON Schema.

For future open-license-only publication, use `--license-policy permissive`; however, publish it as a complete core8 config only after non-NC sources cover all eight categories.

## Load after publication

```python
from datasets import load_dataset

ds = load_dataset(
    "saeeew/JP-HomophoneBench",
    "homophone8-research",
)
exact = ds["test"].filter(lambda row: row["category"] == "exact_homophone")
semantic = ds["test"].filter(lambda row: row["category"] == "semantic_only")
```

## Recommended reporting

Report CER/NE-CER plus per-category Acc@1, Entity Exact Match, MRR, Oracle@4/8/16/32, Bias-FPR, RTF, p50/p95 latency, and VRAM. A high Oracle@K with poor Acc@1 indicates scoring/ranking rather than search failure.

See `release-v0.1.0.md` for the frozen release counts and hashes.
