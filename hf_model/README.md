---
language:
  - ja
license: cc-by-4.0
library_name: nemo
pipeline_tag: automatic-speech-recognition
base_model:
  - nvidia/parakeet-tdt_ctc-0.6b-ja
datasets:
  - saeeew/JP-HomophoneBench
tags:
  - japanese
  - asr
  - nemo
  - tdt
  - ctc
  - contextual-asr
  - context-biasing
  - homophone
  - phoneme
  - proper-noun
  - jpacf
  - yomi
---

# J-PACF-YOMI-TDT

**J-PACF-YOMI-TDT** is the model/method family for this repository's research on Japanese contextual ASR.

- **J-PACF**: Japanese Phoneme-Aware Context Fusion
- **YOMI**: pronunciation/reading-aware contextual disambiguation for Japanese
- **TDT**: Token-and-Duration Transducer decoding based on `nvidia/parakeet-tdt_ctc-0.6b-ja`

> **Current status:** research scaffold. A standalone fine-tuned `.nemo` checkpoint is **not published yet**. The current project evaluates a frozen Parakeet acoustic model with external/fusion scorers first, and only trains the lightweight phoneme head in E05. This Hub repository is therefore the canonical place for model cards, fusion contracts, scorer artifacts, validated checkpoints, and release manifests as the experiments mature.

## Research goal

Japanese contextual ASR has several failure modes that whole-utterance CER hides:

- exact homophones whose pronunciations are identical but written forms differ;
- near-homophones and pronunciation confusions;
- voicing, long-vowel, geminate, and moraic-nasal confusions;
- proper nouns and user/domain vocabulary;
- pitch-accent and context-only disambiguation.

J-PACF-YOMI-TDT separates these failure modes and combines complementary scorers rather than immediately retraining the full ASR backbone.

```text
                         FastConformer
                              |
                     shared encoder H
                              |
               +--------------+--------------+
               |                             |
               v                             v
              TDT                           CTC
               |                             |
          beam search                  local evidence
               |
       +-------+--------+
       |                |
    NGPU-LM          GPU-PB
       |          / TurboBias-style
       |                |
       |          candidate gate
       |                |
       +--------+-------+
                |
        local CTC scorer
                |
        phoneme scorer
                |
       contextual scorer
                |
                v
         fused hypothesis
```

A general experimental score is:

\[
S(h)=S_{\mathrm{TDT}}(h)
+\alpha S_{\mathrm{LM}}(h)
+\beta S_{\mathrm{PB}}(h)
+I_{\mathrm{entity}}\left[
\gamma S_{\mathrm{CTC-local}}(h)
+\delta S_{\mathrm{phone}}(h)
+\epsilon S_{\mathrm{context}}(h)
\right].
\]

The expensive local acoustic/phoneme/context scorers are intended to be gated by active contextual candidates rather than evaluated over every token and every entity.

## Experiment ladder

| ID | Decoder / scorer | Full ASR retraining |
|---|---|---:|
| E00 | TDT greedy | No |
| E01 | TDT beam | No |
| E02 | + NGPU-LM / KenLM | No |
| E03 | + GPU Phrase Boosting | No |
| E04 | + local CTC N-best reranking | No |
| E05 | + frozen-encoder phoneme CTC head | **phoneme head only** |
| E06 | scorer integration inside TDT beam | Depends on final implementation |

Current NeMo `malsd_batch` decoding uses GPU Phrase Boosting keys under `rnnt_decoding.malsd.boosting_tree.*`, while NGPU-LM configuration remains under `rnnt_decoding.beam.*`.

## Benchmark

The primary benchmark is [`saeeew/JP-HomophoneBench`](https://huggingface.co/datasets/saeeew/JP-HomophoneBench).

Its core8 taxonomy is:

1. `exact_homophone`
2. `near_homophone`
3. `voicing`
4. `long_vowel`
5. `geminate`
6. `moraic_nasal`
7. `pitch_accent`
8. `semantic_only`

Two configs are intentionally separated:

- `homophone8`: public configuration excluding explicit NonCommercial source rows;
- `homophone8-research`: larger mixed-license research configuration. Check row-level source licenses before reuse.

## TurboBias / GPU-PB saturation analysis

The repository includes an explicit diminishing-return test for phrase boosting. Adjacent bias-strength settings are compared with paired bootstrap confidence intervals and exact McNemar tests. The default rule flags saturation when entity benefit has plateaued while contextual false-positive risk rises significantly.

This is used to decide whether stronger phrase boosting is enough, or whether the next experiment should add phoneme/acoustic/entity-context discrimination.

## Planned artifact layout

```text
README.md
fusion_config.yaml
research_manifest.json
artifacts/
  <release>/
    J-PACF-YOMI-TDT.nemo        # only after a validated model artifact exists
    phone_head.pt               # optional E05 scorer
    metrics.json
    saturation.json
    release_manifest.json
```

Until a validated checkpoint is published, **do not interpret this Hub repository as a drop-in replacement for the NVIDIA base model**.

## Reproducibility

Source code and workflows are maintained at:

- `https://github.com/yokane/Parakeet-contextual-fusion-research-jp`

Base acoustic model:

- `nvidia/parakeet-tdt_ctc-0.6b-ja`

Benchmark:

- `saeeew/JP-HomophoneBench`

The GitHub repository records the exact experiment stage, benchmark config/split, Git SHA, NeMo/PyTorch/CUDA environment, and category-aware metrics for GPU runs.

## Licensing

The **research source code** in the GitHub repository is Apache-2.0.

The NVIDIA base model `nvidia/parakeet-tdt_ctc-0.6b-ja` is distributed under **CC-BY-4.0**. Model artifacts in this Hub repository that derive from or package that model are therefore documented under **CC-BY-4.0**, with attribution to the NVIDIA base model. Dataset rows retain their own source-level licensing/provenance in `JP-HomophoneBench`.

## Citation

This project is still an active research implementation. A paper citation will be added when a stable method/checkpoint is released. Until then, please cite the GitHub repository, this model repository, the NVIDIA Parakeet base model, and the original datasets relevant to the benchmark rows you use.
