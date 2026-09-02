# Codex Cloud development environment

このリポジトリは、Codex Cloudでは **CPU/static development** を行い、CUDA/NeMoを必要とするresearch validationは既存のGHCR/Vast/Hugging Face/GitHub Actionsへ委譲する。

## Codex Cloudの設定

Codex environment settingsで次を設定する。

```text
Repository:
  yokane/Parakeet-contextual-fusion-research-jp

Setup script:
  bash scripts/codex/setup.sh

Maintenance script:
  bash scripts/codex/maintenance.sh

Agent internet access:
  Off (recommended default)
```

Codex Cloudのsetup scriptはInternet access付きで実行され、その後agent phaseではInternet accessを無効にできる。このリポジトリではsetup中にmiseとlocked dependencyをmaterializeするため、通常のcode edit / lint / testではagent internet accessを必要としない。

## なぜmiseを使うか

Codexの`universal` imageにはPythonやuvなど一般的なtoolが含まれるが、このプロジェクトは次を厳密に固定している。

```text
Python 3.12.3
uv 0.12.1
Linux x86_64
```

そのためhost側のPython/uvを直接正とせず、`mise.toml` / `mise.lock`を正とする。

Setupは次を行う。

```text
miseの存在確認（なければ公式installer）
 -> mise.tomlをtrust
 -> mise install --locked
 -> uv sync --locked --extra dev
 -> isolated HF transport sync
 -> scripts/codex/preflight.sh
```

Maintenance scriptはcached Codex environmentの再利用時に同じlocked stateへ追随させる。

## Codex agentの標準フロー

作業開始前:

```bash
bash scripts/codex/preflight.sh
```

作業後:

```bash
bash scripts/codex/check.sh
```

`check.sh`は`mise run ci`を実行し、canonical CPU/static CIと同じlint/test/compile/shell syntax/lock/SPDX検証を行う。

## Hugging Face

標準のCodex Cloud開発では`HF_TOKEN`を要求しない。

Codex Cloudのsecretはsetup scriptから利用できるが、agent phase開始前に除去される。このため、secretへ`HF_TOKEN`を設定してagent自身にHub publishをさせる設計にはしない。

プロジェクトの公開identityは次の通り。

```text
Model:     saeeew/J-PACF-YOMI-tdt
Dataset:   saeeew/JP-HomophoneBench
Bucket:    saeeew/J-PACF-YOMI-tdt-bucket
```

通常のCodex taskではGit内のlock/provenance/configを編集・検証する。Model/Dataset/Bucketへのpublishは既存のauthenticated Actions/provider workflowを使用する。

public Hugging Face Hubをagent phaseから直接調査する必要があるtaskだけ、Codex environmentのInternet accessを一時的にlimitedへ変更する。通常はOffを維持する。

## GPU境界

Codex Cloud上では次をauthoritative validationとして扱わない。

```text
CUDA availability
PyTorch CUDA 13.2 execution
NeMo GPU decode
E00-E06 GPU result
Vast GPU behavior
```

これらはportable GHCR runtimeとprovider workflowで検証する。

Codex Cloudから安全に行えるものは次。

```text
Python implementation
unit tests
schema / metrics code
Dockerfile static changes
GitHub Actions workflow changes
Vast/HF orchestration code changes
documentation
lock/reproducibility contract checks
```

GPU-facing変更を行った場合、CodexはCPU/static testを通した上で「GPU validationは外部workflowが必要」と明記する。

## ローカルでCodex Cloud相当を確認する

OpenAIのreference `codex-universal` imageを使う場合でも、project setupは同じscriptを利用できる。

```bash
bash scripts/codex/setup.sh
bash scripts/codex/check.sh
```

setup script自体はCodex固有環境変数へ依存せず、Linux x86_64の通常のCI/containerでも利用できるようにしている。
