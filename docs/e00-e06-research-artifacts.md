# E00-E06 research artifacts and execution playbook

This document is the operational contract for developing and evaluating E00-E06 without repeatedly rebuilding CUDA/NeMo, recompiling KenLM, or regenerating large research artifacts.

The machine-readable counterpart is [`configs/research/e00-e06-artifacts.yaml`](../configs/research/e00-e06-artifacts.yaml). Before renting a GPU, validate a phase with:

```bash
uv run --locked --no-sync python scripts/research/check_phase_artifacts.py E04 \
  --state-root /workspace/state
```

## 1. Core policy

Use three storage/execution planes and do not mix their responsibilities:

```text
GitHub-hosted CPU                       Vast GPU
----------------                       --------
benchmark/audio materialization        model/tokenizer-bound work
KenLM estimation                       E00/E01 decoding
phone-head training/reranking           E02 NGPU-LM packing + decoding
static validation                       E03 phrase biasing
                                       E04 CTC reranking
                                       E05 encoder extraction
                                       E06 in-beam integration
          \                            /
           \                          /
            Hugging Face Bucket
            reusable research state
            workspace-cache/e00-e06/<research-key>/
```

Container registries contain **software environments only**. Research data, audio, encoder tensors, language models, results and evidence do not belong in image layers.

The canonical Bucket is:

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket
```

The public benchmark and model identities remain:

```text
saeeew/JP-HomophoneBench
saeeew/J-PACF-YOMI-tdt
```

The exact revisions used by an experiment come from `locks/hf-revisions.lock.json`.

## 2. Deterministic research workspace key

The reusable state key is derived from the locked benchmark revision, locked base-model revision and N-gram order:

```bash
uv run --locked --no-sync python scripts/research/research_key.py
```

Example shape:

```text
v1-bench-<12hex>-model-<12hex>-ng6
```

Remote reusable state:

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/
└── workspace-cache/
    └── e00-e06/
        └── <research-key>/
            ├── generated/
            │   ├── eval/
            │   └── phone_train.jsonl
            ├── artifacts/
            │   ├── lm/
            │   ├── encoder/
            │   ├── encoder_train/
            │   ├── phone_vocab.json
            │   └── phone_head.pt
            └── results/
```

This prefix is deliberately mutable/reusable. Immutable experiment evidence continues to be published separately below `runs/<run-id>/`.

## 3. Image layout

### 3.1 Heavy parent: one copy only

All GPU phase images inherit from the authoritative portable runtime:

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime@sha256:<digest>
```

That parent owns:

- Linux/amd64
- CUDA 13
- Python 3.12.3
- uv 0.12.1
- torch 2.12.0+cu132
- NeMo 3.0.0
- repository runtime environment
- isolated HF Bucket transport environment

Do not reinstall these dependencies in E00-E06 images.

### 3.2 Thin phase tags

Canonical published tags are:

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e00-<git-sha>
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e01-<git-sha>
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e02-<git-sha>
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e03-<git-sha>
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e04-<git-sha>
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e05-<git-sha>
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e06-<git-sha>
```

`*-current` aliases are moved only by an intentional main/manual publication.

E00/E01/E03/E04/E05/E06 are targets in `docker/phases/Dockerfile` and add only dispatch/contract files to the heavy parent.

E02 is built from `docker/research/Dockerfile.e02`. It adds only the pinned KenLM runtime binaries and E02 orchestration script to the same heavy parent. The KenLM source tree and compiler toolchain are not copied into the E02 GPU image.

### 3.3 CPU research-tool images

Two environments are intentionally independent from CUDA:

```text
ghcr.io/yokane/jpacf-yomi-tdt-tools:kenlm-<kenlm-revision-short>
ghcr.io/yokane/jpacf-yomi-tdt-tools:phone-e05-<git-sha>
```

`kenlm-*` contains only the pinned `lmplz`/`build_binary` toolchain plus minimal runtime libraries and Python for metadata validation.

`phone-e05-*` contains Python 3.12 + CPU PyTorch + the small E05 phone-head scripts. It does not contain CUDA, NeMo or the 0.6B ASR checkpoint.

## 4. Why E02 is split into three preparation stages

NeMo's transducer N-gram path is not a plain Japanese word-level KenLM. For a BPE/subword ASR model, NeMo first maps tokenizer IDs to Unicode symbols using its token offset, trains KenLM over those symbols, and then can package the ARPA into `NGramGPULanguageModel` form.

Therefore E02 preparation is split by actual dependency boundary:

```text
lm_corpus.txt
   |
   | Vast / exact NeMo + locked Parakeet tokenizer
   v
lm_corpus.encoded.txt + encoding-metadata.json
   |
   | GitHub-hosted CPU / pinned KenLM container
   v
ja-6gram.arpa + ja-6gram.binary + estimation-metadata.json
   |
   | Vast / exact NeMo 3.0 runtime
   v
ja-6gram.nemo + package-metadata.json
   |
   v
E02/E03/E04/E06
```

This keeps the expensive `lmplz` estimation on GitHub-hosted CPU while avoiding a multi-GB CUDA runtime pull on that runner just to access NeMo's tokenizer/NGPU packaging code.

The implementation is `scripts/research/ngram_lm_pipeline.py`:

```bash
# Vast image: exact model tokenizer -> encoded corpus
python scripts/research/ngram_lm_pipeline.py encode ...

# GitHub-hosted + KenLM tools image
python3 scripts/research/ngram_lm_pipeline.py estimate ...

# Vast image: ARPA -> NGramGPULanguageModel .nemo
python scripts/research/ngram_lm_pipeline.py pack ...
```

The KenLM revision remains pinned to:

```text
4cb443e60b7bf2c0ddf3c745378f76cb59e254e5
```

## 5. Why E05 is split

E05 has one expensive model-bound operation and several small tensor operations:

```text
E04_ctc_rerank.jsonl + nemo_eval.jsonl
         |
         | Vast / locked Parakeet encoder
         v
artifacts/encoder/*.pt
         |
         | GitHub-hosted CPU
         +--> prepare_phone_head_data.py
         +--> train_phone_head.py --device cpu
         +--> rerank_phone.py --device cpu
         v
phone_vocab.json
phone_head.pt
E04_phone_ready.jsonl
E05_phone_rerank.jsonl
```

The canonical split route is:

- `e05-extract` on `research-phase-vast.yml`
- `e05-phone` on `research-artifacts-cpu.yml`

`run-phase.sh E05` remains available as a monolithic debugging route, but it is not the cost-optimized canonical route.

## 6. Common artifacts: GitHub-hosted CPU

Run the `research-artifacts-cpu` workflow with task `common`.

It executes `scripts/research/prepare_common_artifacts.sh` and produces:

| Artifact | Path under state root | Used by |
|---|---|---|
| benchmark index | `generated/eval/bench_index.jsonl` | E05/metrics/evidence |
| execution manifest | `generated/eval/nemo_eval.jsonl` | E00-E04/E06 |
| audio | `generated/eval/audio/` | GPU decoding |
| provenance | `generated/eval/eval_provenance.json` | audit/evidence |
| audio coverage | `generated/eval/audio_coverage.json` | gate |
| context phrases | `generated/eval/context_phrases.txt` | E03/E04/E06 |
| LM corpus | `generated/eval/lm_corpus.txt` | E02 preparation |

When the state is restored on Vast, `scripts/research/rebase_eval_manifest.py` rewrites the provider-local absolute audio paths to `/workspace/state/generated/eval/audio/...`.

## 7. Per-phase artifact contracts

### E00 — TDT greedy baseline

Executor: **Vast GPU**

Required before launch:

```text
generated/eval/nemo_eval.jsonl
generated/eval/audio/*
locked base-model revision
```

Output:

```text
results/E00_tdt_greedy.jsonl
```

Purpose: establish the no-beam/no-LM/no-context baseline.

### E01 — TDT MAES/beam

Executor: **Vast GPU**

Required:

```text
generated/eval/nemo_eval.jsonl
```

Output:

```text
results/E01_tdt_beam.jsonl
```

Keep beam-size changes recorded in run evidence. E01 isolates search gain before external LM/context biasing.

### E02 — NGPU-LM

Preparation executors:

1. `e02-encode`: **Vast**
2. `e02-estimate`: **GitHub-hosted CPU**
3. `e02-pack`: **Vast**
4. `E02`: **Vast**

Artifacts:

```text
# Common input
generated/eval/lm_corpus.txt

# Vast tokenizer stage
artifacts/lm/lm_corpus.encoded.txt
artifacts/lm/encoding-metadata.json

# Hosted KenLM stage
artifacts/lm/ja-6gram.arpa
artifacts/lm/ja-6gram.binary
artifacts/lm/estimation-metadata.json

# Vast NeMo packaging stage
artifacts/lm/ja-6gram.nemo
artifacts/lm/package-metadata.json

# Experiment output
results/E02_ngpulm.jsonl
```

The `.binary` is retained as portable KenLM evidence; the `.nemo` artifact is the canonical input for the repository's NeMo GPU decoder path.

### E03 — GPU phrase/context biasing

Executor: **Vast GPU**

Required:

```text
generated/eval/nemo_eval.jsonl
generated/eval/context_phrases.txt
artifacts/lm/ja-6gram.nemo
```

Output:

```text
results/E03_gpu_pb.jsonl
```

No dedicated dependency image is needed. Current NeMo word-boosting/boosting-tree support is already part of the authoritative runtime.

### E04 — local hybrid-CTC N-best rerank

Executor: **Vast GPU**

Required:

```text
generated/eval/nemo_eval.jsonl
generated/eval/context_phrases.txt
artifacts/lm/ja-6gram.nemo
```

Outputs:

```text
results/E04_nbest.jsonl
results/E04_ctc_rerank.jsonl
```

The CTC branch runs the locked 0.6B model again, so E04 remains a GPU task.

### E05 — frozen-encoder phoneme CTC rerank

Executors: **Vast + GitHub-hosted CPU**

Vast input:

```text
generated/eval/nemo_eval.jsonl
```

Vast output:

```text
artifacts/encoder/*.pt
```

Hosted CPU inputs:

```text
generated/eval/bench_index.jsonl
results/E04_ctc_rerank.jsonl
artifacts/encoder/*.pt
```

Hosted CPU outputs:

```text
artifacts/encoder_train/*.pt
generated/phone_train.jsonl
artifacts/phone_vocab.json
artifacts/phone_head.pt
results/E04_phone_ready.jsonl
results/E05_phone_rerank.jsonl
```

### E06 — version-isolated in-beam fusion

Executor: **Vast GPU only**

Required:

```text
generated/eval/nemo_eval.jsonl
generated/eval/context_phrases.txt
artifacts/lm/ja-6gram.nemo
E06_DRIVER=<pinned NeMo-3.0.0-specific driver>
```

Output:

```text
results/E06_inbeam.jsonl
```

There is intentionally no independent patched E06 base image yet because the repository does not currently contain a promoted `patches/nemo-<sha>/inbeam_driver.py`. Once a driver satisfies the promotion gate in `patches/README.md`, create a thin driver overlay on `phase-e06` rather than creating another CUDA/NeMo base.

## 8. Recommended research sequence

The sequence below minimizes idle GPU time and allows independent lanes to overlap.

```text
A. common (GitHub-hosted)
   |
   +---------------------------> E00 (Vast)
   +---------------------------> E01 (Vast)
   |
   +--> e02-encode (Vast, short)
          |
          v
       e02-estimate (GitHub-hosted)
          |
          v
       e02-pack (Vast, short)
          |
          v
       E02 -> E03 -> E04 (Vast)
                         |
                         v
                 e05-extract (Vast)
                         |
                         v
                 e05-phone (GitHub-hosted)
                         |
                         v
                        E05
                         |
                   promotion gate
                         |
                         v
                        E06 (Vast)
```

A practical manual workflow sequence is:

```text
research-artifacts-cpu: task=common
research-phase-vast:    task=E00
research-phase-vast:    task=E01
research-phase-vast:    task=e02-encode
research-artifacts-cpu: task=e02-estimate
research-phase-vast:    task=e02-pack
research-phase-vast:    task=E02
research-phase-vast:    task=E03
research-phase-vast:    task=E04
research-phase-vast:    task=e05-extract
research-artifacts-cpu: task=e05-phone
# E06 only after the E04/E05 promotion criterion is met.
```

All workflow invocations must use the same `research_key`. Leaving it blank derives the same key from the locked revisions.

## 9. Build and publish research images

The `research-images` workflow builds/publishes the software environments on GitHub-hosted CPU.

It uses a `docker-container` Buildx builder and pushes directly to the registry. It does **not** use `--load`, and it deliberately does **not** export Docker layer state to the GitHub Actions cache.

Why:

- the huge CUDA/NeMo parent already exists as an immutable registry object;
- phase overlays are tiny;
- KenLM is pinned and tagged by revision, so an existing immutable tag is a stronger cache than repeatedly exporting build cache;
- GitHub Actions cache should stay reserved for mise/uv and small build artifacts;
- source/audio/model/encoder tensors never enter Docker layers.

The builder first checks whether an immutable tag already exists. If it exists, the image is reused without rebuilding.

Local equivalent:

```bash
export SOURCE_SHA="$(git rev-parse HEAD)"
export SOURCE_REPOSITORY="https://github.com/yokane/Parakeet-contextual-fusion-research-jp"
export RUNTIME_IMAGE="ghcr.io/yokane/jpacf-yomi-tdt-runtime:runtime-current"
export GHCR_PHASE_REPO="ghcr.io/yokane/jpacf-yomi-tdt-runtime"
export GHCR_TOOLS_REPO="ghcr.io/yokane/jpacf-yomi-tdt-tools"
bash scripts/ci/build_research_images.sh
```

`dist/research-images.json` records the exact registry, digest and resolved parent image.

## 10. GHCR pressure and Docker Hub fallback

GHCR is primary because E00-E06 can reuse the existing runtime blobs. The research-image workflow does not create separate heavyweight base images per phase.

Optional repository secrets:

```text
DOCKERHUB_ACCESS_TOKEN
DOCKERHUB_REPOSITORY
```

`DOCKERHUB_REPOSITORY` is expected to identify one **public** Docker Hub repository, for example:

```text
namespace/jpacf-yomi-tdt-research
```

The username is derived from the namespace. Do not store a Docker Hub password; use the access token.

Publication policy:

1. try immutable GHCR tag;
2. if it already exists, reuse it;
3. otherwise build with direct Buildx registry push;
4. if GHCR push fails and Docker Hub fallback is configured, retry the same build to:

```text
docker.io/<DOCKERHUB_REPOSITORY>:<same-immutable-suffix>
```

5. record the actual registry/digest in `dist/research-images.json`;
6. Vast accepts an explicit image override, so a Docker Hub fallback image can be used without changing the artifact contract.

Do not mirror every successful GHCR image to Docker Hub by default. Mirroring would duplicate multi-GB CUDA parent blobs and defeat the storage objective.

## 11. Cache/storage rules

### GHCR / Docker Hub

Allowed:

```text
software layers
pinned KenLM binaries
small phase dispatch scripts
small CPU phone-head tool runtime
OCI labels/manifests
```

Forbidden:

```text
JP-HomophoneBench audio
.nemo base model checkpoint copied as research data
KenLM trained language-model artifact
encoder *.pt tensors
phone_head.pt
experiment result JSONL/Parquet
HF run evidence
```

### Hugging Face Bucket

Use for:

```text
rehydrated evaluation audio
portable manifests
encoded LM corpus
ARPA/binary/NGPU language-model artifacts
encoder features
phone-head artifacts
mutable reusable research state
append-only run evidence
```

`hf buckets sync` transfers only changed files and supports plan/apply. Use it instead of archiving the whole state into GitHub Actions cache.

### GitHub Actions cache

Use only for existing small tool/download caches such as mise/uv. Do not add Docker `type=gha,mode=max` for these thin research images.

### GitHub workflow artifacts

Use only for short-lived control-plane evidence such as:

```text
research-images.json
Vast selected-offer JSON
Vast create response/status/log
```

The research workflows use 14-day retention for these small diagnostics.

## 12. Phase readiness checks

Examples:

```bash
STATE=/workspace/state
uv run --locked --no-sync python scripts/research/check_phase_artifacts.py E00 --state-root "$STATE"
uv run --locked --no-sync python scripts/research/check_phase_artifacts.py E02 --state-root "$STATE"
uv run --locked --no-sync python scripts/research/check_phase_artifacts.py E05 --state-root "$STATE"
```

A failed readiness check is a preparation failure, not a GPU experiment failure. Fix the producer task first rather than renting another Vast instance.

## 13. Reproducibility metadata to preserve

Every reusable artifact should be attributable to:

- benchmark repo + full revision;
- base model repo + full revision;
- source Git SHA;
- phase image digest;
- KenLM revision and N-gram order for E02;
- corpus SHA-256, ARPA SHA-256 and binary SHA-256 for E02;
- NeMo 3.0.0 and torch 2.12.0+cu132 for GPU work;
- E05 phone-head hyperparameters and encoder-feature model revision;
- Vast offer/GPU/cost for GPU evidence;
- `research_key` and immutable HF `runs/<run-id>` evidence path.

The E02 pipeline writes encoding/estimation/package metadata JSON specifically so an LM artifact cannot silently drift away from its corpus/tokenizer/revision.

## 14. Upstream design references

The implementation follows the current upstream contracts reviewed during this work:

- Docker Buildx registry exporter and registry cache documentation: direct `--push` avoids loading the image into the local Docker Engine, and registry/image exporters can publish by digest/tag.
- Docker multi-stage builds: named targets share their parent layers instead of duplicating a complete environment.
- KenLM: CMake build, `lmplz`, and `build_binary` are the canonical estimation tools.
- NVIDIA NeMo Speech: transducer/TDT beam decoding accepts KenLM/NGPU-LM inputs; `malsd_batch` word boosting uses the boosting-tree configuration.
- Hugging Face Buckets: `hf buckets sync` works local-to-bucket and bucket-to-local and supports plan/apply for controlled synchronization.

The repository pins the actual versions/revisions it validates; upstream examples are design references, not substitutes for the repository locks.
