# E00-E06 phase images

E00-E06 can be materialized as separate Docker images without rebuilding the CUDA/PyTorch/NeMo dependency rootfs. The phase Dockerfile uses named multi-stage targets that all inherit one authoritative portable runtime digest.

## Build contract

```text
portable GHCR runtime @ sha256:...
        |
        +-- target e00 -> E00 greedy
        +-- target e01 -> E01 beam
        +-- target e02 -> E02 N-gram LM
        +-- target e03 -> E03 context biasing
        +-- target e04 -> E04 CTC rerank
        +-- target e05 -> E05 phone-head preparation + rerank
        +-- target e06 -> E06 version-specific in-beam driver
```

The Dockerfile is `docker/phases/Dockerfile`.

Always resolve the parent runtime to an immutable digest before building a phase image:

```bash
export JPA_CF_IMAGE="$(
  bash scripts/container/resolve-image.sh \
    ghcr.io/yokane/jpacf-yomi-tdt-runtime:main
)"
```

Build one phase locally:

```bash
docker buildx build \
  --file docker/phases/Dockerfile \
  --target e03 \
  --build-arg "RUNTIME_IMAGE=${JPA_CF_IMAGE}" \
  --build-arg "SOURCE_REVISION=$(git rev-parse HEAD)" \
  --tag jpacf:e03 \
  --load \
  .
```

For CI/provider use, prefer `--push` instead of `--load` so the large parent image is not exported into a hosted runner's local Docker image store.

## Runtime state

Mount one persistent state directory at `/workspace/state`:

```bash
docker run --rm --gpus all \
  -v "$HOME/.cache/jpacf-state:/workspace/state" \
  jpacf:e03
```

The inherited `scripts/container/inside.sh` maps the important locations into this state tree:

```text
/workspace/state/generated/eval   benchmark/manifests/context lists
/workspace/state/artifacts        LM, phone head, encoder features
/workspace/state/results          E00-E06 outputs
/workspace/state/hf               HF Hub/Xet cache
```

Cross-provider `artifacts/` and `generated/` can be restored from the deterministic HF Bucket `workspace-cache/<key>` before running a phase.

## Phase prerequisites

| Target | Phase | Main prerequisites |
|---|---|---|
| `e00` | TDT greedy | `nemo_eval.jsonl` |
| `e01` | TDT beam | `nemo_eval.jsonl` |
| `e02` | + N-gram LM | manifest + `ja-6gram.nemo` |
| `e03` | + context biasing | E02 prerequisites + context phrases |
| `e04` | + CTC rerank | E03 prerequisites |
| `e05` | + phoneme CTC rerank | E04 result + benchmark index; encoder features/phone head are prepared by `run-phase.sh` by default |
| `e06` | in-beam fusion | E03 prerequisites + `E06_DRIVER` compatible with pinned NeMo 3.0.0 |

E05 supports `E05_PREPARE=0` when precomputed phone artifacts already exist. E06 intentionally remains version-isolated and refuses to run without `E06_DRIVER`.

## Hugging Face portability

The public benchmark remains metadata-first, so phase images do not bake benchmark audio into image layers. Rehydrate/materialize the evaluation data into `/workspace/state/generated/eval`, or restore the deterministic workspace cache from the project HF Bucket.

This keeps the phase image reusable on WSL2, self-hosted runners, Vast.ai, and container-based Hugging Face Jobs while the large mutable research data remains outside the image.
