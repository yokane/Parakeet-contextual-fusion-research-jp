# Local development with Dev Containers

This directory provides the lightweight local development environment for the repository.

The default devcontainer deliberately matches the repository's **CPU/static CI lane**, not the multi-GB CUDA/NeMo research runtime. This keeps normal editing, linting, tests, benchmark tooling, and Hugging Face transport fast while preserving the canonical GPU image as the single source of truth for E00-E06 execution.

## Requirements

- Docker Desktop, Docker Engine, or an equivalent Dev Containers-compatible runtime
- VS Code/Cursor with Dev Containers support
- Enough disk for the Python/uv environment and local Hugging Face data

The repository lock is Linux/x86_64-only. The Dockerfile therefore builds an `linux/amd64` development image. On ARM hosts this uses Docker's amd64 emulation and will be slower.

## Open the repository

1. Clone the repository normally.
2. Open the repository root in VS Code/Cursor.
3. Run **Dev Containers: Reopen in Container**.
4. Wait for `.devcontainer/post-create.sh` to finish.

The bootstrap performs:

```text
mise 2026.8.16
  -> mise.lock
     -> Python 3.12.3
     -> uv 0.12.1
        -> uv sync --locked --extra dev
        -> isolated tools/hf-bucket uv.lock sync
```

It does not install the `gpu` extra.

## Persistent local state

Docker named volumes are used so rebuilding the devcontainer does not throw away every package/download cache:

```text
jpacf-dev-mise   -> /home/vscode/.local/share/mise
jpacf-dev-cache  -> /home/vscode/.cache
jpacf-dev-state  -> /workspace/state
```

`/workspace/state` is intentionally separate from the source checkout and follows the same state separation used by the canonical runtime.

To completely reset the local development state, remove those Docker volumes explicitly.

## Common commands

Canonical CPU/static validation:

```bash
mise run ci
```

Individual checks:

```bash
mise run lint
mise run test
mise run compile
mise run locks:verify
mise run spdx:check
```

Refresh only the locked CPU development environment:

```bash
mise run deps:sync
```

Refresh the isolated Hugging Face Bucket transport environment:

```bash
mise run hf:transport:sync
```

Run benchmark/data tooling from the same container, for example:

```bash
make bench-permissive
make bench-validate
```

For operations that need Hugging Face authentication, export the token only in your interactive shell or inject it through your normal secret-management mechanism. Do not commit a token into `devcontainer.json`, `.env`, or repository files.

```bash
export HF_TOKEN=hf_...
make hf-eval-index
```

## GPU research remains containerized separately

The default devcontainer is not the authoritative E00-E06 GPU environment. GPU validation continues to use the repository's canonical image:

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime
```

This avoids maintaining two independent NeMo/CUDA environments.

For local NVIDIA execution, leave the editor in the lightweight devcontainer and run the canonical runtime from a host terminal using the existing repository wrapper, for example:

```bash
mkdir -p .jpacf-state
JPA_CF_IMAGE=ghcr.io/yokane/jpacf-yomi-tdt-runtime:runtime-current \
  bash scripts/container/run.sh
```

For a reproducible experiment, replace `runtime-current` with the exact digest recorded by the corresponding runtime/image manifest.

## Dependency changes

Code-only changes are immediately available in the source checkout. If `pyproject.toml`, `uv.lock`, CUDA, PyTorch, or NeMo dependencies change:

- rebuild the devcontainer for CPU/static work;
- rebuild and GPU-verify the canonical runtime before treating the new GPU dependency set as valid.

The devcontainer must not become a second authoritative GPU runtime.
