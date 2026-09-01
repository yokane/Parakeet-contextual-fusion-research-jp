# J-PACF Hugging Face / GHCR storage policy

## Roles

J-PACF-YOMI-TDT separates source, execution environment, work artifacts, and accepted releases.

```text
GitHub source
    │
    ├── build/cache ───────► GHCR runtime + BuildKit cache
    │
    ├── GPU experiments ──► HF Bucket runs/
    │
    └── release bundle ───► HF Bucket candidates/
                                  │
                         validation/promotion
                                  │
                                  ▼
                         HF Model Repo artifacts/
```

Canonical targets are source-controlled in `configs/hf-storage.json`:

```text
work bucket:  saeeew/J-PACF-YOMI-tdt-bucket
model repo:   saeeew/J-PACF-YOMI-tdt
benchmark:    saeeew/JP-HomophoneBench
base model:   nvidia/parakeet-tdt_ctc-0.6b-ja
runtime:      ghcr.io/yokane/jpacf-yomi-tdt-runtime
```

The Bucket is development/evaluation storage. It is not accepted release history. The Model Repo contains only promoted model/scorer/config releases.

## Bucket layout

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/
├── README.md
├── config/
│   ├── README.md
│   ├── current.json
│   └── versions/
│       └── config-NNNNNN/
├── candidates/
│   └── candidate-NNNNNN/
├── experiments/
│   └── experiment-NNNNNN/
├── runs/
│   └── <run-id>/
│       ├── run-context.json
│       ├── samples.jsonl
│       ├── metrics.json
│       ├── run.parquet
│       ├── evidence/
│       └── promotion.json   # only after promotion
├── benchmarks/
├── reference/
├── scripts/
└── tmp/
```

`hf-bucket-maintenance.yml` can bootstrap missing root markers without replacing existing objects and can validate the canonical layout.

## Mutability policy

Mutable pointers/status:

```text
README.md
config/current.json
```

Immutable by policy:

```text
config/versions/config-NNNNNN/
candidates/candidate-NNNNNN/
experiments/experiment-NNNNNN/
runs/<run-id>/
benchmarks/.../<run-id>.json
artifacts/<release>/ in the HF Model Repo
```

Do not use Bucket deletion as normal cleanup for candidates or runs. A published candidate/run receives a new identity when it must be regenerated.

## Central allocation

Candidate, experiment, and config IDs use six-digit monotonic IDs:

```text
candidate-000001
experiment-000001
config-000001
```

`.github/workflows/hf-central-allocator.yml` is serialized with one workflow-level concurrency lock. It recursively lists the Bucket, finds the maximum historical/canonical suffix, reserves the next ID by creating `README.md`, and returns the allocation through a short-lived GitHub Actions artifact.

This prevents two concurrent producers from publishing the same ID.

## Candidate publication

A candidate starts as a release-shaped local directory produced by `build_hf_model_release.py` and `prepare_candidate.py`.

```text
release bundle
   │
   ├── release_manifest.json + SHA-256
   ├── fusion_config.yaml
   ├── research_manifest.json
   ├── optional phone_head.pt
   ├── optional checkpoint.nemo
   └── optional metrics / saturation / environment
   │
   ▼
validate candidate
   │
central allocator
   │
   ▼
candidate-NNNNNN
   │
   ▼
hf buckets sync --plan
   │
validate_sync_plan.py
   │
   ├── upload only: accept
   └── download/delete/skip/unknown: reject
   │
   ▼
hf buckets sync --apply
```

A candidate fetched from the Bucket receives `.candidate-id`. `hf-push-candidate.sh` refuses to republish such a materialized candidate as a new identity.

## Run publication

Successful GPU evaluation workflows package durable execution evidence with `build_run_bundle.py` and upload it automatically to `runs/`.

Current run producers:

```text
homophone-eval-gpu
homophone-saturation-gpu
homophone-context-stress-gpu
```

Run IDs include the GitHub run ID and attempt, for example:

```text
gh-123456789-1-eval
gh-123456789-1-saturation
gh-123456789-1-context-stress
```

`hf-push-run.sh` fails when any object already exists under the run ID and validates an upload-only sync plan before applying it.

GitHub Actions artifacts remain short-lived convenience copies; HF Bucket `runs/` is the durable experimental evidence store.

## Promotion

Promotion is Bucket-first:

```text
candidate-NNNNNN
      │
 fresh download
      │
 SHA / manifest validation
      │
 optional .nemo restore validation
      │
      ▼
saeeew/J-PACF-YOMI-tdt/artifacts/<release>/
      │
      └── promotion.json -> originating Bucket run (optional)
```

`hf-model-artifact-publish.yml` no longer accepts arbitrary local model paths. It accepts a Bucket candidate ID and re-fetches it before promotion.

`publish_hf_model_release.py` rejects an already-existing `artifacts/<release>/` prefix. Accepted release names are therefore immutable by policy even though the Hub repository itself has Git history.

## Docker cache boundary

The Dockerfile follows dependency-first layering:

```text
base image
  │
apt packages                  <- BuildKit apt cache
  │
COPY pyproject.toml
  │
install Python + NeMo deps    <- BuildKit pip cache
  │
COPY source/config/scripts
  │
pip install --no-deps -e .
```

Source-only edits therefore invalidate only the final inexpensive layers. Dependency installation is invalidated only when dependency metadata changes.

The build context excludes generated audio, model weights, experiment outputs, local caches, and release artifacts through `.dockerignore`.

## GitHub Actions cache policy

CPU CI caches the installed `.venv` with a key based on:

```text
runner OS
runner architecture
Python version
pyproject.toml hash
```

The source tree is deliberately excluded from the key. Every run performs only the cheap source-only editable install after restoring dependencies.

Self-hosted GPU jobs cache pip downloads. Large model/audio caches are not copied into GitHub Actions cache because self-hosted runner storage is persistent and multi-gigabyte Actions cache churn is counterproductive.

## GHCR policy

`ghcr-runtime.yml` builds:

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime:main
ghcr.io/yokane/jpacf-yomi-tdt-runtime:latest
ghcr.io/yokane/jpacf-yomi-tdt-runtime:sha-<full-git-sha>
```

BuildKit imports from both:

```text
GitHub Actions cache
GHCR :buildcache
```

and exports `mode=max` to both. This allows hosted CI, self-hosted builders, and external Docker-capable compute to reuse dependency layers without rebuilding NeMo/Python dependencies after source-only changes.

Use the full-SHA image for reproducible experiment execution. `main` and `latest` are convenience pointers.

## Commands

```bash
# Verify/initialize Bucket structure
make hf-bucket-validate
make hf-bucket-bootstrap

# Publish a prepared candidate
make hf-candidate-push CANDIDATE_DIR=dist/hf-candidate/<release>

# Publish a prepared run bundle
make hf-run-push RUN_DIR=dist/hf-runs/<run-id>

# Local BuildKit build
make docker-build

# Pull shared runtime
make docker-pull
```

All HF write commands require `HF_TOKEN`. Candidate allocation additionally requires GitHub access because the central allocator is the sequence authority.
