# GHCR Build Optimization, Vast.ai Verification, and Hugging Face Storage Architecture

This document records the architecture, failure modes, fixes, and operating rules introduced while making the J-PACF-YOMI-TDT GPU runtime portable across GitHub Actions, WSL2/self-hosted runners, and Vast.ai.

It complements [`portable-gpu-runtime.md`](portable-gpu-runtime.md): that guide explains how to use the final runtime; this document explains why the runtime is built and operated this way.

The final architecture was introduced by PR #10, `feat(gpu): make GHCR the portable research runtime`.

## 1. Goals

The research environment contains a large CUDA/PyTorch/NeMo dependency root filesystem. The implementation therefore optimizes for reproducibility, portability, and bounded provider cost rather than merely producing a Docker image.

The main goals are:

1. use one authoritative GPU user-space environment everywhere;
2. avoid reinstalling Python/PyTorch/NeMo on every GPU host;
3. avoid rebuilding or re-exporting the CUDA/NeMo rootfs for ordinary source commits;
4. identify runtimes by immutable digest instead of mutable tag;
5. verify the exact published digest on a real GPU;
6. separate immutable software from mutable research state;
7. make important state transferable between WSL2, self-hosted runners, and Vast hosts;
8. keep provider-local caches disposable;
9. record enough evidence to reproduce source, runtime, GPU, and storage identity;
10. always destroy paid Vast instances after proof collection.

```text
Git source
   |
   v
GHCR dependency base
   |
   v
GHCR thin runtime
   |
   +--> /workspace/state
   |      fast provider-local state
   |
   +--> HF Bucket
          transferable state and evidence
```

The storage rule is:

```text
GHCR image       = immutable environment
/workspace/state = fast local working set
HF Bucket        = cross-provider transferable state/evidence
```

## 2. Two-layer Docker architecture

The runtime is split into:

```text
Dockerfile.runtime-base  -> expensive dependency-sensitive layer
Dockerfile               -> source-sensitive thin runtime layer
```

`Dockerfile.runtime-base` owns the expensive pieces:

- NVIDIA CUDA base image;
- system build dependencies;
- Python 3.12.3;
- uv 0.12.1;
- torch 2.12.0+cu132;
- NeMo 3.0.0;
- locked Python dependencies;
- isolated Hugging Face Bucket transport tooling.

The normal `Dockerfile` starts from the exact dependency-base digest and copies source-controlled project material such as `src/`, `scripts/`, `schemas/`, `configs/`, `experiments/`, `hf_model/`, and `locks/`.

This prevents an ordinary source edit from invalidating the CUDA/NeMo installation layers.

### Dependency-base key

`scripts/ci/build_runtime_image.sh` hashes only files capable of changing the expensive dependency rootfs:

```text
Dockerfile.runtime-base
.dockerignore
pyproject.toml
uv.lock
locks/containers.lock.json
tools/hf-bucket/**
```

The resulting tag is:

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime:base-<base-key>
```

CI checks for the tag with `docker buildx imagetools inspect`. If it exists, the dependency image is reused rather than rebuilt.

A helper tag, `base-current`, is used as a cache source. It is not the authoritative runtime identity.

### Digest chaining

The dependency tag is resolved to an immutable reference:

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime@sha256:<base-digest>
```

That exact reference is supplied as `BASE_IMAGE` to the thin runtime build.

The runtime is published under source-oriented and cache-helper tags and then resolved to an immutable digest. Authoritative GPU proof and experiments consume the digest form.

## 3. Direct-to-registry Buildx

### Failure that motivated the change

The hosted build was able to finish the expensive Docker stages, but failed while exporting the very large final image into the hosted runner's local Docker Engine image store.

```text
Docker stages complete
      |
      v
large local image export/unpack
      |
      v
hosted runner shutdown / exit 143
```

This was an image-store I/O problem, not a Python or NeMo dependency failure.

### Final solution

The hosted fallback workflow configures Buildx with the `docker-container` driver and the build script uses `--push`.

The canonical hosted path deliberately does not use `--load`.

With the `docker-container` driver, build results are not automatically loaded into the local Docker Engine image store. `--push` exports the result directly to GHCR. This removes the failure path where the multi-gigabyte CUDA/NeMo rootfs had to be materialized in the hosted runner's Docker daemon after the build.

After publishing, the workflow validates the remote manifest with:

```bash
docker buildx imagetools inspect <runtime-digest-reference>
```

It does not pull the image back into the same hosted runner merely to validate it. Real execution validation happens later on a GPU host.

## 4. Build caches

Both dependency and thin-runtime builds export inline cache metadata. Mutable helper tags such as `base-current` and `runtime-current` can be used as remote cache sources.

The dependency Dockerfile also uses BuildKit cache mounts for apt and uv downloads. For apt, `sharing=locked` prevents parallel builds from concurrently modifying apt cache/database state.

Cache state is an optimization only. A clean build must remain correct if caches are missing, overwritten, or garbage-collected.

### Remaining optimization opportunity

The thin `Dockerfile` still performs a final locked `uv sync` after source has been copied. A source change can therefore invalidate this final layer. Even if dependencies are unchanged, BuildKit may still need to materialize the large parent snapshot to execute that instruction.

This is safer than the former local `--load` path, but it is not the theoretical minimum I/O. A future optimization can investigate keeping source importable through `PYTHONPATH=/opt/jpacf/src` or another mechanism that avoids mutating the environment for source-only changes while preserving the locked dependency contract.

## 5. Immutable Python versus mutable state

An early configuration put build-time user state under `/workspace/state/home`. uv-managed Python could then be installed below mutable user state, causing the project venv interpreter to resolve into a directory hidden later by the `/workspace/state` mount.

The dependency image now explicitly uses:

```text
UV_PROJECT_ENVIRONMENT=/opt/jpacf/.venv
UV_PYTHON_INSTALL_DIR=/opt/jpacf/.uv-python
UV_PYTHON_INSTALL_BIN=0
```

The image build verifies that `/opt/jpacf/.venv/bin/python` resolves below `/opt/jpacf/.uv-python/`.

Only runtime-writable data is redirected to `/workspace/state`, including `HOME`, Hugging Face caches, uv cache, XDG cache, torch cache, artifacts, generated data, results, and distribution evidence.

The invariant is:

```text
/opt/jpacf        immutable executable environment
/workspace/state  mutable cache/result state
```

Never mount provider state over `/opt/jpacf`.

## 6. UID/GID handling on self-hosted runners

A verification container that runs as root can create root-owned directories in persistent `.jpacf-state`. A later staged workflow running as the runner user may then be unable to write them.

The shared container wrapper therefore uses the host UID/GID for normal research execution. GPU verification and HF publication paths use the common wrapper rather than ad-hoc root containers.

This keeps persistent state writable across repeated self-hosted runs.

## 7. Vast.ai as an execution target

Vast is not treated as the source of the Python/NeMo environment. It runs the exact GHCR digest directly.

Vast supplies:

```text
GPU
NVIDIA host driver/runtime
local disk
network
container execution
```

The GHCR image supplies:

```text
Python
PyTorch
CUDA user-space
NeMo
research code
HF Bucket transport tooling
```

The canonical paid fallback workflow is `.github/workflows/ghcr-runtime-vast-fallback.yml` and is manual (`workflow_dispatch`) by design.

Its flow is:

```text
build
  -> publish exact runtime digest to GHCR

verify-vast
  -> choose budget-compliant RTX 4090 offer
  -> create instance using exact digest
  -> wait for GPU proof marker
  -> write runtime identity
  -> publish evidence to HF Bucket
  -> destroy instance
  -> upload short-lived GitHub artifact
```

## 8. Vast offer selection

The current canonical policy targets:

```text
GPU              RTX 4090
GPU RAM          >= 24 GB
internet down    >= 1000 Mbps
reliability      >= 0.98
storage          80 GB
predicted window 30 minutes
max predicted    0.35 USD
```

The workflow ranks offers with repository scripts. High image download bandwidth matters because a fresh Vast host may spend more time pulling and extracting the large CUDA/NeMo image than running the actual CUDA probe.

For short jobs, a slightly higher hourly price can be cheaper in total wall-clock cost if registry throughput is substantially better.

## 9. Vast argv/shell quoting failure

An early provider command attempted to pass a nested shell program through Vast `--args`. The runtime entrypoint ultimately executes argv directly, so Vast delivered the quoted shell expression as a single executable argument.

The container then attempted to execute a pathname containing the whole `bash -lc ...` string and exited before GPU verification.

The image itself was valid; the provider argv boundary was wrong.

The fix is `scripts/container/vast_verify.sh`, passed as one absolute executable:

```text
--args /opt/jpacf/scripts/container/vast_verify.sh
```

The verifier script:

```text
runs verify_runtime.py --require-gpu
captures the return code
prints JPA_CF_CANONICAL_VERIFY rc=<code>
exits immediately on failure
keeps the container alive only after successful proof
```

This removes nested shell quoting from the Vast API/CLI/container boundary.

## 10. Vast fail-fast and cleanup

A paid verification must not wait for a long workflow timeout after the container has already failed.

The current wait loop:

1. polls raw instance status;
2. records `actual_status`;
3. fetches logs with command-level timeouts;
4. succeeds only after `JPA_CF_CANONICAL_VERIFY rc=0`;
5. fails immediately on a non-zero proof marker;
6. fails immediately if the instance is `exited` before success;
7. is bounded by an outer attempt count.

The destroy step runs with `always()` semantics and destroys the recorded instance ID even if earlier verification or publication steps fail.

Provider CLI operations are also individually bounded with `timeout` so a hung CLI call cannot indefinitely bypass cleanup.

## 11. GPU verification contract

A successful proof is intentionally stronger than `nvidia-smi`.

The verifier checks the software/runtime contract and performs a real CUDA tensor computation.

Expected properties include:

```text
platform             linux/amd64
Python               3.12.3
Python executable    /opt/jpacf/.venv/bin/python
torch                2.12.0+cu132
torch compiled CUDA  13.2
NeMo                 3.0.0
CUDA available       true
GPU count            >= 1
state writable       hf/uv/xdg/torch = true
```

A canonical Vast proof after the final fixes recorded:

```text
source SHA
  5c9c79856093eded02f67dfe148f56e99f607a9b

dependency base digest
  sha256:277bd7a4bfd7dcec97d4afecb21c71af15a939b91bf03078525a490d3df08725

runtime digest
  sha256:8deb1301693f28ebf6e9373e47c81adb7f2eade0edc4303abfddb14ecb6bee09

GitHub Actions run
  33571017033

Vast offer
  40232085

Vast instance
  49592872

GPU
  NVIDIA GeForce RTX 4090

observed hourly price
  0.674074074074074 USD/hour

predicted 30-minute cost at selection time
  0.337 USD

GitHub evidence artifact
  9825427447
```

The proof emitted `JPA_CF_CANONICAL_VERIFY rc=0`. The instance was destroyed after HF publication and evidence upload.

These values are historical proof data, not permanent configuration. Future authoritative runs must record their own digest and provider metadata.

## 12. Hugging Face responsibilities

GHCR and Hugging Face have separate roles:

```text
GHCR      executable immutable environment
HF Bucket transferable files, caches, identities, and experiment evidence
```

The project does not copy the full GPU root filesystem through HF Bucket. Conversely, important evidence should not exist only inside an ephemeral provider disk.

### Isolated HF transport environment

HF Bucket transport is isolated under `/opt/jpacf/tools/hf-bucket`.

`scripts/hf/hf-identity.sh` exposes `hf_bucket_cli`. Inside the authoritative container, it runs the pre-materialized transport project with locked `--no-sync` behavior so an experiment upload cannot silently resynchronize or mutate the ASR environment.

Outside the canonical container, the helper may synchronize the transport project normally.

## 13. HF Bucket namespace

The project bucket is:

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket
```

The logical layout is:

```text
runtime/sha-<git-sha>/
  verified runtime identity and provider proof

workspace-cache/<deterministic-key>/
  artifacts/
  generated/

runs/<run-id>/
  immutable experiment evidence
```

This separates runtime identity, reusable materialized research state, and experiment results.

## 14. Runtime identity publication

After GPU proof, the workflow writes `runtime-image.json` with source SHA, platform, CUDA/PyTorch/NeMo contract, dependency-base digest, runtime digest, execution contract, state mount, GPU verification status, and provider metadata where relevant.

The verified identity is synchronized to:

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/runtime/sha-<source-sha>
```

For the canonical proof above:

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/runtime/sha-5c9c79856093eded02f67dfe148f56e99f607a9b
```

This gives a provider-independent lookup from source revision to the runtime digest that actually passed GPU verification.

## 15. Cross-provider workspace cache

Not every cache should be transferred between providers.

The project transfers expensive reproducible research material:

```text
artifacts/
generated/
```

Disposable package/download caches stay provider-local:

```text
hf/
uv/
xdg/
torch/
```

A deterministic key is generated by `scripts/container/cache-key.sh`. Remote data is stored under:

```text
hf://buckets/<bucket>/workspace-cache/<key>
```

Published cache keys are treated as immutable. If relevant inputs change, a new key is produced instead of overwriting an existing cache.

`scripts/hf/hf-sync-workspace-cache.sh` uses a plan/apply publication pattern for `artifacts/` and `generated/` and refuses to overwrite an already published key.

This makes the workspace cache behave more like content-addressed research material than a mutable network directory.

## 16. Provider-local persistence versus HF Bucket

A Vast Volume or self-hosted `.jpacf-state` remains useful for speed, but it is not the cross-provider source of truth.

Provider-local state is optimized for repeated access on the same host and can contain large disposable caches. HF Bucket is optimized for portability, deterministic reusable material, runtime identity, and immutable evidence.

The intended strategy is:

```text
local state first for speed
+
HF Bucket for portability and evidence
```

## 17. End-to-end canonical flow

```text
1. read pinned NVIDIA base digest
2. calculate dependency-base key
3. reuse existing base or build once
4. resolve dependency base to immutable digest
5. build thin source runtime
6. push directly from Buildx to GHCR
7. resolve exact runtime digest
8. select GPU target
9. run exact digest on real NVIDIA GPU
10. execute CUDA tensor proof
11. record runtime identity
12. sync identity/evidence to HF Bucket
13. destroy paid Vast instance when applicable
14. use verified digest for experiments
```

Experiment execution then restores deterministic HF workspace material when available, prepares missing artifacts, runs the experiment ladder, and publishes immutable run evidence.

## 18. Required GitHub secrets

The orchestration paths use repository secrets named:

```text
VAST_API_KEY
HF_TOKEN
```

GitHub's workflow token is used for GHCR operations with package permissions.

Secret values must never be committed into repository files, Docker layers, build metadata, or diagnostic artifacts.

## 19. Operational commands

Resolve the promoted runtime to an immutable digest:

```bash
export JPA_CF_IMAGE="$(
  bash scripts/container/resolve-image.sh \
    ghcr.io/yokane/jpacf-yomi-tdt-runtime:main
)"
```

Verify on local/self-hosted Linux with NVIDIA Docker support:

```bash
bash scripts/container/run.sh \
  python /opt/jpacf/scripts/container/verify_runtime.py --require-gpu
```

Generate a deterministic workspace cache key:

```bash
KEY="$(bash scripts/container/cache-key.sh)"
```

Restore workspace material:

```bash
bash scripts/container/run.sh \
  bash scripts/hf/hf-sync-workspace-cache.sh pull "$KEY"
```

Publish a new immutable workspace cache:

```bash
bash scripts/container/run.sh \
  bash scripts/hf/hf-sync-workspace-cache.sh push "$KEY"
```

Paid canonical Vast verification is performed through the manual `ghcr-runtime-vast-fallback` GitHub Actions workflow.

## 20. Troubleshooting matrix

| Symptom | Likely layer | First checks |
|---|---|---|
| Docker stages finish, runner dies during export | hosted Docker image-store I/O | confirm `docker-container` Buildx and `--push`; reject `--load` |
| dependency base rebuilds after ordinary source edit | base-key scope | inspect `base_key` inputs in `build_runtime_image.sh` |
| project Python disappears after mounting state | interpreter placed under mutable state | verify `/opt/jpacf/.uv-python` invariant |
| self-hosted staged run cannot write persistent state | root-owned files | use the shared wrapper and host UID/GID |
| Vast exits with a path containing an entire shell command | provider argv quoting | pass `vast_verify.sh` as a single executable path |
| Vast waits after container already died | polling not fail-fast | inspect `actual_status` and proof-marker handling |
| Vast keeps charging after failure | cleanup path | verify persisted instance ID and `always()` destroy step |
| HF upload unexpectedly changes ASR environment | transport env not isolated | use `hf_bucket_cli` and locked container transport |
| a new provider lacks materialized artifacts | local volume treated as portable | restore deterministic HF workspace cache |
| tag and evidence disagree | mutable tag used as identity | resolve and record `@sha256:` before execution |

## 21. Architecture invariants

Future changes should preserve these rules:

1. Do not put managed Python under `/workspace/state`.
2. Do not mount provider state over `/opt/jpacf`.
3. Do not use `--load` in the large hosted canonical build path.
4. Do not promote a runtime solely because it built successfully; verify the exact digest on a real GPU.
5. Do not use mutable tags as experiment identity.
6. Do not let ad-hoc root containers create persistent self-hosted research state.
7. Do not pass complicated quoted shell programs through the Vast `--args` boundary.
8. Do not make paid Vast cleanup depend on success of earlier steps.
9. Do not resynchronize HF transport dependencies into the locked ASR runtime during research execution.
10. Do not overwrite deterministic HF workspace-cache keys.
11. Do not treat a Vast Volume as the cross-provider source of truth.
12. Do not pull a just-pushed huge hosted image back into the same runner merely to validate its manifest.

## 22. Upstream documentation used to validate the design

The implementation was checked against current upstream documentation through Context7.

Docker:

- <https://docs.docker.com/build/builders/drivers/>
- <https://docs.docker.com/build/exporters/image-registry/>
- <https://docs.docker.com/build/cache/backends/registry/>
- <https://docs.docker.com/build/cache/optimize/>

Relevant Docker behavior confirmed by the current documentation:

- `docker-container` does not automatically load results into the local Docker image store;
- `--push` exports the result to a registry;
- `--load` is the explicit local-image-store path for non-default builders;
- cache exporters/importers can be used independently of immutable image identity;
- `RUN --mount=type=cache` is a performance optimization and must not be required for runtime correctness.

Hugging Face Hub:

- <https://huggingface.co/docs/huggingface_hub/guides/cli>
- <https://huggingface.co/docs/huggingface_hub/guides/buckets>

Relevant HF behavior confirmed by the current documentation:

- bucket sync supports local-to-bucket and bucket-to-local directory synchronization;
- synchronization can use a `--plan` and `--apply` two-phase flow;
- authentication is supplied separately from the data being synchronized;
- normal sync transfers only the files that require an operation.

## 23. Related repository files

Core build/runtime:

```text
Dockerfile.runtime-base
Dockerfile
scripts/ci/build_runtime_image.sh
scripts/container/run.sh
scripts/container/inside.sh
scripts/container/resolve-image.sh
scripts/container/verify_runtime.py
scripts/container/vast_verify.sh
```

Vast orchestration:

```text
.github/workflows/ghcr-runtime-vast-fallback.yml
scripts/providers/vast/build_search_query.py
scripts/providers/vast/rank_offers.py
```

Hugging Face storage:

```text
configs/hf-storage.json
scripts/hf/hf-identity.sh
scripts/hf/hf-sync-workspace-cache.sh
tools/hf-bucket/
```

Usage guide:

```text
docs/portable-gpu-runtime.md
```

Regression contracts:

```text
tests/test_container_contract.py
tests/test_runtime_managed_python_contract.py
tests/test_hf_bucket_storage.py
tests/test_self_hosted_cache_contract.py
```

When this architecture changes, update both the implementation and the regression tests so these invariants remain machine-checkable.
