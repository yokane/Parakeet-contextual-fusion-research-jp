# J-PACF-YOMI-TDT model release lifecycle

The Hugging Face model repository is `saeeew/J-PACF-YOMI-tdt`.

The GitHub repository remains the source of truth for implementation and experiments. The Hugging Face model repository is the distribution surface for the model card, canonical fusion configuration, release manifests, lightweight scorer artifacts, and eventually validated `.nemo` checkpoints.

## Why the base model is not copied by default

E00-E04 keep `nvidia/parakeet-tdt_ctc-0.6b-ja` frozen. In those stages, the research contribution is decoding/fusion configuration rather than a new 0.6B acoustic checkpoint. Copying the NVIDIA `.nemo` weights into every experimental release would:

- duplicate large files;
- make it harder to distinguish a new model from a decoder configuration;
- obscure which weights actually changed;
- complicate attribution and release review.

Therefore the default release is a **scorer/config release**.

## Release types

### 1. Scaffold

Published from `.github/workflows/hf-model-publish.yml`.

Contains the root model card and canonical contracts:

```text
README.md
fusion_config.yaml
research_manifest.json
```

This workflow is manual-only after the initial repository bootstrap.

### 2. Scorer/config release

Used when the acoustic model remains frozen. Typical contents:

```text
artifacts/v0.x.y-e05/
├── fusion_config.yaml
├── research_manifest.json
├── phone_head.pt
├── metrics.json
├── saturation.json
├── environment.json
├── RELEASE_NOTES.md
└── release_manifest.json
```

`release_manifest.json` records the source Git SHA and SHA-256/size for every copied artifact.

### 3. Standalone checkpoint release

Only use this mode when the research actually produces a validated `.nemo` model artifact.

Before upload, `.github/workflows/hf-model-artifact-publish.yml` restores the checkpoint using:

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.restore_from(
    checkpoint_path,
    map_location="cpu",
)
```

If restore fails, publication fails.

## Manual artifact publication

Run the GitHub Actions workflow:

```text
hf-model-artifact-publish
```

It requires the self-hosted runner labels:

```text
self-hosted
linux
gpu
```

Inputs are local paths on that runner. Only `release_name` is required. The remaining paths are optional:

- `checkpoint_path`
- `phone_head_path`
- `metrics_path`
- `saturation_path`
- `environment_path`
- `notes_path`

A scorer-only E05 release can therefore publish only a phone head and its evidence, while a future integrated model can additionally publish `J-PACF-YOMI-TDT.nemo`.

## Hub metadata policy

At the current research-scaffold stage, the Model Card YAML deliberately does not set `base_model` or `datasets`.

Reason:

- Hugging Face interprets `base_model` as a derived-model relationship such as fine-tune, adapter, quantized, or merge.
- `JP-HomophoneBench` is currently an evaluation benchmark, not the training dataset of a new acoustic model.

The runtime dependency and benchmark are still recorded explicitly in the human-readable Model Card, `fusion_config.yaml`, and `research_manifest.json`.

When a genuinely derived standalone model is published, update the Model Card with the appropriate `base_model` and `base_model_relation` at that time.

## License boundary

- GitHub source code: Apache-2.0.
- NVIDIA Parakeet base model: CC-BY-4.0.
- Any artifact containing or deriving from NVIDIA base weights must preserve the required attribution/license terms.
- Dataset rows retain row-level provenance/license information in `saeeew/JP-HomophoneBench`.

The model repository is currently documented under CC-BY-4.0 so a future base-derived checkpoint is not accidentally advertised under the Apache-2.0 source-code license.

## Release gate

A release should be published only after the relevant evidence exists:

| Release | Minimum gate |
|---|---|
| decoder config | E00-E04 reproducible metrics |
| phrase-boost config | saturation sweep + FPR report |
| phoneme head | E05 category-specific gain + no unacceptable general-CER regression |
| integrated `.nemo` | successful NeMo restore + E00-E06 reproducibility + release manifest |

For phoneme-scoring claims, report near-homophone/voicing/long-vowel/geminate/moraic-nasal performance separately from exact-homophone/semantic-only performance. Exact homophones cannot be resolved by segmental pronunciation evidence alone.
