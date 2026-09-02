# Repository instructions for Codex and other coding agents

## Scope

These instructions apply to the entire repository.

This project has two intentionally separate execution environments:

1. **CPU/static development environment** — used by Codex Cloud, ordinary local development, and the canonical CPU CI.
2. **Authoritative GPU research environment** — the pinned GHCR CUDA 13 / PyTorch / NeMo runtime used by self-hosted GPU runners, Vast.ai, and provider workflows.

Do not try to turn the Codex Cloud container into the authoritative GPU runtime.

## First commands

Before modifying code, run:

```bash
bash scripts/codex/preflight.sh
```

After modifications, run:

```bash
bash scripts/codex/check.sh
```

`check.sh` executes the repository's canonical CPU/static CI through the locked mise environment.

## Toolchain contract

- Linux x86_64 is the supported development platform.
- Python is exactly `3.12.3`.
- uv is exactly `0.12.1`.
- `mise.toml`, `mise.lock`, `pyproject.toml`, and `uv.lock` are authoritative.
- Prefer `mise run <task>` or `mise exec -- <command>` over the host Python, pip, or uv.
- Do not refresh or rewrite lockfiles unless the task explicitly asks for a dependency/lock update.
- Do not add the `gpu` extra to the Codex Cloud environment. GPU validity is checked outside Codex Cloud.

## Validation

The default validation command is:

```bash
bash scripts/codex/check.sh
```

For focused iteration, these tasks are available:

```bash
mise run lint
mise run test
mise run compile
mise run shell:syntax
mise run locks:verify
mise run spdx:check
```

If you change Docker/GHCR/Vast/Hugging Face GPU orchestration, run all CPU/static contract tests locally and clearly state which GPU/provider validation still needs an external workflow.

## Hugging Face rules

Canonical repositories are defined in `mise.toml` and lock files. Current project identities are:

- model: `saeeew/J-PACF-YOMI-tdt`
- benchmark: `saeeew/JP-HomophoneBench`
- bucket: `saeeew/J-PACF-YOMI-tdt-bucket`

The benchmark is metadata-first. Do not add downloaded source audio, model weights, generated manifests, or provider caches to Git.

Standard Codex Cloud development must not require `HF_TOKEN`. Do not print, persist, or commit tokens. Publishing models, datasets, runtime evidence, or Bucket contents belongs in the existing authenticated GitHub Actions/provider workflows unless the user explicitly requests another publication path.

## GPU research boundary

E00-E06 execution that requires CUDA/NeMo belongs in the portable runtime documented under `docs/portable-gpu-runtime.md` and `docs/e00-e06-phase-images.md`.

Codex Cloud may safely modify and statically validate GPU-facing code, Dockerfiles, provider scripts, schemas, and workflows, but it must not claim real GPU validation unless an actual GPU workflow/provider run produced the evidence.

## Repository hygiene

Do not commit generated or local state such as:

- `.venv/`
- `.jpacf-state/`
- `artifacts/`
- `results/`
- `data/generated/`
- model/checkpoint files (`*.nemo`, `*.pt`)

Use `git diff --check` before finishing. Keep changes narrowly scoped and preserve immutable revision/digest contracts unless the task explicitly changes them.
