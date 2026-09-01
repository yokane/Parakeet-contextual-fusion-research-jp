# Portable GHCR GPU runtime

This repository treats the GHCR image as the authoritative GPU execution environment. The host is deliberately thin: it supplies a Linux x86_64 Docker runtime, an NVIDIA GPU/driver, and storage. Python, PyTorch, NeMo, CUDA user-space libraries, the Hugging Face Bucket transport environment, and the research scripts are pinned in the image.

## Contract

```text
Git repository / source checkout
  editable code and experiment definitions
              |
              v
GHCR runtime image (immutable digest)
  Python 3.12.3
  uv 0.12.1
  torch 2.12.0+cu132
  NeMo 3.0.0
  CUDA 13 user-space
  HF Bucket CLI environment
              |
              +---- /workspace/project   optional bind mount of current source
              |
              +---- /workspace/state     one persistent state volume
                         |
                         +-- hf/          Hub/Xet cache
                         +-- uv/          uv download cache
                         +-- xdg/         library caches
                         +-- torch/       torch cache
                         +-- home/        disposable container HOME state
                         +-- artifacts/   model/LM/phone-head artifacts
                         +-- generated/   materialized benchmark/audio/manifests
                         +-- results/     experiment results
                         +-- dist/        bundles/runtime identity
                         +-- vendor/      pinned NeMo Speech/KenLM source/builds

HF Bucket
  runtime/sha-<git-sha>/                 verified image identity
  workspace-cache/<deterministic-key>/   cross-host reusable artifacts/generated
  runs/<run-id>/                         immutable experiment evidence
```

The image is the environment; `/workspace/state` is the fast local working set; the HF Bucket is the transferable state between hosts/providers. A Vast Volume is therefore a cache for one Vast host, not the cross-provider source of truth.

## Image identity

The publishing workflow creates:

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime:sha-<git-sha>
```

and verifies that exact digest with a real CUDA tensor operation before a main-branch image is promoted to:

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime:main
```

For authoritative experiment runs, resolve tags to an immutable digest first:

```bash
export JPA_CF_IMAGE="$(
  bash scripts/container/resolve-image.sh \
    ghcr.io/yokane/jpacf-yomi-tdt-runtime:main
)"

echo "$JPA_CF_IMAGE"
# ghcr.io/yokane/jpacf-yomi-tdt-runtime@sha256:...
```

`runtime-image.json` records the digest, Git SHA, CUDA/PyTorch/NeMo contract, and successful GPU verification. A copy is also published under the HF Bucket `runtime/` namespace.

## WSL2 / local Linux

Keep the persistent state on the Linux filesystem rather than a Windows `/mnt/c` bind when possible.

```bash
cd ~/src/Parakeet-contextual-fusion-research-jp

export JPA_CF_IMAGE="$(
  bash scripts/container/resolve-image.sh \
    ghcr.io/yokane/jpacf-yomi-tdt-runtime:main
)"
export JPA_CF_STATE_ROOT="$HOME/.cache/jpacf-state"
export HF_BUCKET=saeeew/J-PACF-YOMI-tdt-bucket
export HF_TOKEN=hf_...
```

Verify the host/container boundary:

```bash
bash scripts/container/run.sh \
  python /opt/jpacf/scripts/container/verify_runtime.py --require-gpu
```

A successful verification proves more than `nvidia-smi`: PyTorch must initialize CUDA, allocate a tensor on `cuda:0`, execute a computation, synchronize it, and report the device.

Run the staged research ladder:

```bash
export JPA_CF_RESULTS_NAME=selfhosted
export RUN_E05=auto
export RUN_E06=0

bash scripts/container/run.sh bash -lc '
  bash scripts/research/prepare_e00_e04.sh
  bash experiments/run_staged_e00_e06.sh
'
```

The host does not need the repository's Python/NeMo environment for this path.

## Docker Compose development shell

After resolving `JPA_CF_IMAGE` as above:

```bash
export JPA_CF_UID="$(id -u)"
export JPA_CF_GID="$(id -g)"
export JPA_CF_STATE_ROOT="$HOME/.cache/jpacf-state"

docker compose run --rm research \
  python /opt/jpacf/scripts/container/verify_runtime.py --require-gpu

docker compose run --rm research bash
```

The source checkout is mounted at `/workspace/project`; the baked `/opt/jpacf/.venv` remains visible and immutable.

## Cross-provider workspace cache via HF Bucket

The expensive reusable research state is intentionally smaller than a complete filesystem snapshot. It contains `artifacts/` and `generated/`; raw package caches remain local and can be rematerialized from pinned revisions.

Derive a deterministic key from runtime/model/dataset locks and materialization scripts:

```bash
KEY="$(bash scripts/container/cache-key.sh)"
echo "$KEY"
```

Restore before expensive preparation:

```bash
bash scripts/container/run.sh \
  bash scripts/hf/hf-sync-workspace-cache.sh pull "$KEY"
```

If no cache exists, prepare the assets normally and publish a new immutable key:

```bash
bash scripts/container/run.sh bash -lc \
  'bash scripts/research/prepare_e00_e04.sh'

bash scripts/container/run.sh \
  bash scripts/hf/hf-sync-workspace-cache.sh push "$KEY"
```

A published `workspace-cache/<key>` is immutable. Changing the relevant locks or preparation scripts produces a different key rather than mutating an old cache.

## Vast.ai

Vast should run the GHCR image directly. Do not start a generic Vast container and then try to run Docker inside it.

Use a versioned GHCR locator such as:

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime:sha-<git-sha>
```

The corresponding verified digest is recorded in `runtime-image.json` and the HF Bucket runtime identity.

Attach a Vast persistent Volume at exactly:

```text
/workspace/state
```

Do not mount the Volume over `/opt/jpacf`, because that would hide the baked runtime and scripts.

For a validation command/entrypoint, run:

```bash
bash /opt/jpacf/scripts/container/inside.sh \
  python /opt/jpacf/scripts/container/verify_runtime.py --require-gpu
```

For E00-E06:

```bash
bash /opt/jpacf/scripts/container/inside.sh bash -lc '
  cd /opt/jpacf
  bash scripts/research/prepare_e00_e04.sh
  bash experiments/run_staged_e00_e06.sh
'
```

Set environment variables such as:

```text
HF_TOKEN=<secret>
HF_BUCKET=saeeew/J-PACF-YOMI-tdt-bucket
JPA_CF_RESULTS_NAME=vast-<run-name>
RUN_E05=auto
RUN_E06=0
```

A Vast Volume improves restart speed on the same physical host. If the next instance is placed on another Vast host, restore `artifacts/generated` from the deterministic HF Bucket workspace cache instead of assuming the Volume can follow the instance.

## GitHub self-hosted runner

The self-hosted GPU workflow no longer installs or activates host `mise`, Python, PyTorch, NeMo, or CUDA user-space dependencies for authoritative research. Its responsibilities are:

```text
checkout
 -> GHCR login
 -> resolve image tag to digest
 -> docker run --gpus all verifier
 -> run E00-E06 inside that same digest
 -> write state under .jpacf-state/
 -> publish immutable evidence to HF Bucket
 -> upload short-lived GitHub artifact
```

This keeps WSL2 and Vast on the same executable environment contract.

## What remains host-dependent

Containerization intentionally does not hide failures below the container boundary. The following must still work on each host:

```text
NVIDIA GPU/driver
Docker engine
NVIDIA Container Toolkit / Docker GPU integration
filesystem used for /workspace/state
network access to GHCR and Hugging Face
```

If this fails:

```bash
docker run --rm --gpus all \
  ghcr.io/yokane/jpacf-yomi-tdt-runtime@sha256:... \
  bash /opt/jpacf/scripts/container/inside.sh \
  python /opt/jpacf/scripts/container/verify_runtime.py --require-gpu
```

then the problem is below the Python/NeMo environment layer and should be diagnosed as a Docker/NVIDIA host-runtime issue rather than by rebuilding the project virtualenv.
