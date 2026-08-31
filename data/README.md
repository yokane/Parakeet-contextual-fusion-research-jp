# Benchmark data

This repository keeps benchmark metadata, manifests, and small seed files in Git. Do not commit large upstream audio blobs.

## Hugging Face sources

### `NagaYu/mondegreen-asr-errors`

Use as an error-taxonomy and hard-negative source. Its current public build is synthetic text-pair data rather than a measured real-ASR audio corpus, so numbers derived from it must not be reported as acoustic ASR measurements.

Suggested mapping:

- voicing -> `voicing`
- long vowel -> `long_vowel`
- geminate -> `geminate`
- moraic nasal -> `moraic_nasal`
- term-phonetic -> `near_homophone`

### `HaitongSUN/prosodic-abx`

Use Japanese natural-speech minimal pairs for `pitch_accent`. The fixed benchmark stores source row locators and target timing metadata instead of republishing the waveform by default.

### `IDEMITSU/hoiku-yougo-stt-ja`

Use structured `reading`, `mis_conversions`, and context fields to construct exact/near-homophone candidate graphs. The source is CC BY-NC 4.0, so the default `permissive` Hugging Face publication policy excludes rows derived from it.

### `JDSC-Project/SS-JDSC`

Useful as an optional `acoustic_stress` control. It is a specialized difficult-speech corpus, not a representative general-Japanese benchmark.

### `sbintuitions/joyo-kanji-yomi-benchmark`

Useful as a reading/context ambiguity source. It is text/TTS-oriented; any generated ASR audio should be explicitly marked synthetic and reported separately from natural speech.

## Difficulty is multidimensional

Do not collapse difficulty to one ordinal label. Store independent dimensions:

```json
{
  "difficulty": {
    "acoustic": 0.7,
    "lexical": 0.4,
    "context": 0.2,
    "phone_distance": 0.25
  }
}
```

`phone_distance=0` is not necessarily easy. Two different written forms with identical pronunciation can be maximally difficult for segmental-acoustic evidence.

## Generated layout

```text
data/releases/v0.1.0/
├── all.jsonl
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── stats.json

data/generated/
├── nemo_eval.jsonl
├── context_phrases.txt
├── lm_corpus.txt
└── provenance.json
```

Release JSONL is immutable. Create a new versioned directory instead of overwriting a benchmark used in reported experiments.
