# JP-HomophoneBench v0.1.0 release provenance

Published to Hugging Face as:

```text
saeeew/JP-HomophoneBench
config: homophone8-research
visibility: public
license policy: research
```

## Validation

The release completed the following GitHub Actions stages successfully:

1. build deterministic JSONL
2. validate JSON Schema
3. validate split/group leakage invariants
4. validate exact/semantic homophone phone identity
5. validate core8 completeness
6. upload train/validation/test Parquet shards to Hugging Face
7. upload immutable JSONL build artifact

GitHub Actions run: `33448269115`.

## Size

```text
records:    660
groups:     327
train:      439
validation: 69
test:       152
```

## Core8 category counts

```text
exact_homophone  165
near_homophone   200
voicing           15
long_vowel        15
geminate           6
moraic_nasal       2
pitch_accent      92
semantic_only    165
```

`core8_complete = true`.

## Included sources

```text
HaitongSUN/prosodic-abx           92   CC BY 4.0
IDEMITSU/hoiku-yougo-stt-ja      562   CC BY-NC 4.0
manual-cc0-example                 6   CC0 1.0
```

`NagaYu/mondegreen-asr-errors` remains supported by the builder, but no rows from it survived into this frozen v0.1.0 release after the current adapter/filtering/deduplication path. Do not claim it as an included source for v0.1.0.

Because this release contains CC BY-NC-derived rows, it is intentionally published as `homophone8-research`. The dataset is metadata-first and does not redistribute upstream audio.

## Frozen JSONL SHA-256

```text
train.jsonl
  a04d4c3c44fc7001b1e22a72516a4c5ce4b1d020c22a1193c21b88f7b9ff452e

validation.jsonl
  546fd3be40142d05d584d3f76a02608020ba75c580b7f2bbfc10c47c42f3e653

test.jsonl
  37c34098152519172eb65c2033ff5be52546e6b22ff0bb37275fc00d3721331f

all.jsonl
  7496a979fc0404be6b04302fda4555a80eedea9f57c2b8d436a4be094297458b
```

## GitHub Actions artifact

```text
name: jp-homophone-bench-v0.1.0
artifact id: 9778852553
zip size: 141512 bytes
zip sha256: 3b944c17d5a69f0db093265d309851b4e8136b12759e79f37300d05e3c8aae2b
```

This provenance file documents the successful bootstrap publication. Future releases should use a new immutable release directory and config/version rather than modifying the v0.1.0 JSONL fixture in place.
