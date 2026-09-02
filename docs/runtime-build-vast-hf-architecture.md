# GHCR Build最適化・Vast.ai GPU検証・Hugging Faceストレージ設計

このドキュメントは、J-PACF-YOMI-TDT のGPU研究環境を **GitHub Actions / WSL2・self-hosted runner / Vast.ai** で同一条件に近づけるために行った、Docker build最適化、Vast対応、Hugging Face対応の設計判断・失敗事例・修正内容・運用ルールを記録する。

[`portable-gpu-runtime.md`](portable-gpu-runtime.md) が「完成したruntimeをどう使うか」を説明するのに対し、本書は **なぜ現在の構成になったのか** を説明する。

最終構成は PR #10 `feat(gpu): make GHCR the portable research runtime` で `main` に導入された。

---

## 1. 最終的な責務分離

今回の設計では、環境・高速なローカル状態・プロバイダー間で持ち運ぶ状態を明確に分離した。

```text
Git repository
  source / workflow / experiment definitions
              |
              v
GHCR dependency base
  CUDA / Python / PyTorch / NeMo / system dependencies
              |
              v
GHCR thin runtime
  source / scripts / configs / experiments
              |
              +-------------------------------+
              |                               |
              v                               v
/workspace/state                         HF Bucket
高速なprovider-local state              転送可能なstate / evidence
              |                               |
              |                               +-- runtime/sha-<git-sha>/
              |                               +-- workspace-cache/<key>/
              |                               +-- runs/<run-id>/
              |
              +-- hf/
              +-- uv/
              +-- xdg/
              +-- torch/
              +-- artifacts/
              +-- generated/
              +-- results/
              +-- dist/
```

要約すると次の3層になる。

```text
GHCR image       = immutable environment
/workspace/state = fast provider-local working state
HF Bucket        = cross-provider transferable state / evidence
```

Vast Volumeやself-hosted runner上のローカルディスクは高速なキャッシュとして有効だが、プロバイダーを跨ぐsource of truthにはしない。

---

## 2. Docker imageを dependency base と thin runtime に分割した理由

GPU runtimeは次の2つのDockerfileへ分割した。

```text
Dockerfile.runtime-base
  -> 変更頻度が低く、構築コストが高いdependency layer

Dockerfile
  -> 変更頻度が高いsource/runtime layer
```

### `Dockerfile.runtime-base` が持つもの

主に次を固定する。

```text
NVIDIA CUDA base image
system packages / build tools
Python 3.12.3
uv 0.12.1
torch 2.12.0+cu132
NeMo 3.0.0
locked Python dependencies
isolated Hugging Face Bucket transport environment
```

### `Dockerfile` が持つもの

dependency baseの**厳密なdigest**を親にして、source依存部分のみをコピーする。

```text
src/
scripts/
schemas/
configs/
experiments/
hf_model/
locks/
README.md
mise.toml / mise.lock / stack.lock.yaml
```

これにより、通常のPython sourceやexperiment scriptの変更だけでCUDA/NeMoのインストール層を毎回作り直すことを避ける。

---

## 3. dependency base key

`scripts/ci/build_runtime_image.sh` は、高コストなdependency rootfsを変化させうる入力だけから `base_key` を計算する。

対象は次の通り。

```text
Dockerfile.runtime-base
.dockerignore
pyproject.toml
uv.lock
locks/containers.lock.json
tools/hf-bucket/**
```

概念的には以下である。

```text
base_key = sha256(expensive dependency inputs)
```

生成されるimmutable tagは次の形式。

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime:base-<base_key>
```

build前にregistry上の存在を次で確認する。

```bash
docker buildx imagetools inspect "$base_tag"
```

存在すればdependency baseを再buildせず再利用する。

さらにcache source用として次を維持する。

```text
:base-current
```

ただし `base-current` はcache用のmutable tagであり、実験runtimeのidentityには使用しない。

---

## 4. digest chaining

base tagを見つけた、または新規buildした後、必ずdigestを解決する。

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime@sha256:<base-digest>
```

thin runtime buildには、この**digest-pinned reference**を `BASE_IMAGE` として渡す。

そのため、親imageのmutable tagが後から別内容へ移動しても、同じruntime buildが暗黙に異なるdependency rootfsを参照することを防げる。

runtime側は次のtagを作成する。

```text
:sha-<SOURCE_SHA>
:runtime-current
```

そして最終的に次のdigest referenceへ解決する。

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime@sha256:<runtime-digest>
```

**GPU検証とauthoritative experimentはtagではなくdigestを使用する。**

---

## 5. GitHub-hosted runnerで発生したDocker build問題

### 症状

CUDA/NeMoを含むDocker buildそのものは完了していたが、最後の巨大image export時にhosted runnerが終了した。

```text
Docker build stages complete
        |
        v
final imageをlocal Docker image storeへexport / unpack
        |
        v
runner shutdown / exit 143
```

重要なのは、これはPython・PyTorch・NeMoのinstall failureではなく、**巨大imageをrunner側Docker EngineへmaterializeするI/O failure**だった点である。

### 誤った対処になりやすいもの

この症状に対して以下を調整しても本質的な改善にならない。

```text
pip/uv retryを増やす
NeMo versionを変更する
Python buildをやり直す
DockerfileのRUNを細分化するだけ
```

build stage後のexport path自体を変える必要があった。

---

## 6. `docker-container` Buildx + `--push`

hosted fallbackではBuildx driverを次にした。

```yaml
with:
  driver: docker-container
```

そして `build_runtime_image.sh` は最終出力に `--push` を使用する。

```bash
docker buildx build \
  ... \
  --push \
  --tag "$runtime_tag" \
  --tag "$runtime_current"
```

意図的に使用しないものは次。

```text
--load
```

`docker-container` driverではbuild resultは自動的にlocal Docker image storeへloadされない。

`--push` により、BuildKitからGHCRへ直接exportする。

```text
BuildKit
  |
  +---- X ----> hosted Docker Engine local image store
  |
  +-----------> GHCR
```

これにより、以前runner shutdownを起こしていた巨大rootfsのlocal export/unpack経路を排除した。

### build後の検証

push後は次でremote manifestを確認する。

```bash
docker buildx imagetools inspect "$runtime_reference"
```

ここで「検証のために同じhosted runnerへ巨大imageをpullし直す」ことはしない。

imageが実際にGPU上で動くかどうかは、Vastまたはself-hosted GPU runtimeで別途検証する。

---

## 7. BuildKit cacheの使い分け

### registry / inline cache

buildではinline cache metadataをexportする。

```text
--cache-to type=inline
```

cache sourceとしてmutable helper tagを利用する。

```text
:base-current
:runtime-current
```

cacheは高速化のための補助であり、再現性の根拠ではない。

再現性の根拠は以下である。

```text
locked inputs
pinned parent digest
source SHA
published runtime digest
```

### `RUN --mount=type=cache`

`Dockerfile.runtime-base` ではaptとuvでBuildKit cache mountを使用する。

例:

```dockerfile
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update ...
```

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    ...
```

aptには `sharing=locked` を使い、複数buildがapt cache/databaseへ同時書き込みすることを防ぐ。

BuildKit cache mountはperformance optimizationであり、cacheがGC・破損・未生成でもbuild自体が成立する設計にする。

---

## 8. 現時点で残るDocker build最適化余地

高コストdependency installは分離できているが、thin `Dockerfile` はsource copy後に次を実行している。

```text
uv sync --locked --python 3.12.3 --extra dev --extra gpu
```

そのためsource変更によって最終 `RUN` layerがinvalidateされると、dependency自体が変わっていなくてもBuildKitが巨大なparent snapshotを展開してその命令を実行する場合がある。

これは従来の `--load` failureとは別問題であり、現在の構成でもbuild reliabilityは大きく改善している。

将来的な候補は以下。

```text
PYTHONPATH=/opt/jpacf/src を中心にsourceを直接import可能にする
source installとdependency syncをさらに分離する
project metadataだけでinstallできるlayer設計を検討する
```

ただし、現在の「locked environmentを実験中に変化させない」というinvariantを壊してはならない。

---

## 9. `/workspace/state` がPythonを隠した問題

初期実装ではbuild-timeの `HOME` を `/workspace/state/home` に寄せていた。

uv-managed Pythonがuser state配下へinstallされると、`.venv/bin/python` の実体が `/workspace/state/...` 側へ向くことがある。

その状態でruntime時にfresh volumeを `/workspace/state` へmountすると、imageにbakeしたPythonの実体がmountで隠れてしまう。

### 修正後

dependency imageでは以下を明示した。

```text
UV_PROJECT_ENVIRONMENT=/opt/jpacf/.venv
UV_PYTHON_INSTALL_DIR=/opt/jpacf/.uv-python
UV_PYTHON_INSTALL_BIN=0
```

さらにbuild時に以下を検証する。

```bash
readlink -f /opt/jpacf/.venv/bin/python
```

解決先が必ず次の配下であることをassertする。

```text
/opt/jpacf/.uv-python/
```

runtimeに入ってから初めて、write可能なstateを `/workspace/state` へ向ける。

```text
HOME=/workspace/state/home
HF_HOME=/workspace/state/hf
UV_CACHE_DIR=/workspace/state/uv
XDG_CACHE_HOME=/workspace/state/xdg
TORCH_HOME=/workspace/state/torch
```

最終invariantは次。

```text
/opt/jpacf        immutable executable environment
/workspace/state  mutable runtime state
```

**persistent volumeを `/opt/jpacf` へmountしてはいけない。**

---

## 10. self-hosted runnerのUID/GID問題

GPU verificationをraw `docker run` でroot実行すると、persistent `.jpacf-state` 内のdirectoryがroot-ownedになることがあった。

その後のstaged workflowがrunner userで書き込むとpermission failureになる。

現在は共通wrapper `scripts/container/run.sh` を通し、通常のresearch executionではhost UID/GIDを使う。

```text
--user "$(id -u):$(id -g)"
```

GPU verificationとHF publicationも同じwrapperを使うことで、self-hosted runner上のstate ownershipを統一した。

---

## 11. Vast.aiの位置づけ

Vastは「環境を構築する場所」ではなく、**GHCR上のexact runtime digestをGPUで実行する場所**として扱う。

Vast側が提供するもの:

```text
GPU
NVIDIA host driver / container runtime
local disk
network
container execution
```

GHCR image側が提供するもの:

```text
Python
PyTorch
CUDA user-space
NeMo
research source/scripts
HF Bucket transport tooling
```

そのため、Vast instance上で別途mise/Python/NeMoを構築する必要はない。

---

## 12. Canonical Vast fallback workflow

manual workflow:

```text
.github/workflows/ghcr-runtime-vast-fallback.yml
```

現在は `workflow_dispatch` のみで起動する。

有料GPU proofを通常のsource/docs pushで自動発火させないためである。

処理フロー:

```text
build
  -> dependency baseをreuse/build
  -> thin runtimeをGHCRへdirect push
  -> exact runtime digestを出力

verify-vast
  -> offer search
  -> budget/帯域条件でranking
  -> exact digestでVast instance作成
  -> GPU proof marker待機
  -> runtime identity作成
  -> HF Bucketへpublication
  -> Vast instance destroy
  -> GitHub artifact upload
```

---

## 13. Vast offer selection

現在のcanonical policyは次。

```text
GPU              RTX 4090
GPU RAM          >= 24 GB
internet down    >= 1000 Mbps
reliability      >= 0.98
storage          80 GB
predicted window 30 minutes
max predicted    0.35 USD
```

単純な最低価格ではなく、特に `inet_down` を重視する。

理由はruntime imageが大きく、fresh Vast hostではGPU計算よりもGHCRからのpull/extract時間の方が支配的になりうるため。

短時間verificationでは、極端に安いがnetworkが遅いhostより、若干hourly priceが高くても高速download可能なhostの方がwall-clockと総コストの双方で有利になることがある。

---

## 14. Vast `--args` shell quoting failure

初期のVast proofでは、`--args` に概念的に次のようなshell expressionを渡した。

```text
bash -lc "python ...; rc=$?; ..."
```

しかしruntime entrypointは最終的にargvをそのまま `exec` する。

Vast CLI/API境界でquoted shell commandが1つのargumentとして渡されたため、containerはshellを起動するのではなく、**shell expression全体を1つのexecutable pathとして実行しようとした**。

結果:

```text
... bash -lc '...' : No such file or directory
```

Docker image自体は正常で、provider argv contractの問題だった。

### 修正

専用の実行可能scriptを追加した。

```text
scripts/container/vast_verify.sh
```

Vastへ渡すのは1つのabsolute executableのみ。

```text
--args /opt/jpacf/scripts/container/vast_verify.sh
```

scriptは次を行う。

```text
verify_runtime.py --require-gpu を実行
return codeを取得
JPA_CF_CANONICAL_VERIFY rc=<rc> を出力
失敗なら即exit
成功時のみevidence回収のためcontainerを維持
```

これによりVast API/CLI境界からnested shell quotingを排除した。

workflowはinstance作成前に `vast_verify.sh` がexecutableであることも検証する。

---

## 15. Vast fail-fastと課金停止

有料providerでは「workflow timeoutまで待てば安全」では不十分である。

現在のwait loopは次の順で判定する。

1. `vastai show instance --raw` でstatus取得
2. `actual_status` をevidenceへ記録
3. container log取得
4. `JPA_CF_CANONICAL_VERIFY rc=0` なら成功
5. non-zero markerなら即失敗
6. marker未出力のまま `actual_status == exited` なら即失敗
7. bounded retry countを超えたら失敗

さらにprovider CLI自体を `timeout` で囲み、Vast CLIのhangでworkflow全体がcleanupへ進めなくなることを防ぐ。

instance destroy stepは次の意味を持つ。

```yaml
if: ${{ always() }}
```

verificationやHF publicationが失敗しても、保存済みinstance IDに対してdestroyを試行する。

---

## 16. GPU verification contract

成功条件は `nvidia-smi` が表示できることではない。

container内からPyTorchでCUDAを初期化し、GPU tensorを確保して実際のcomputeを行い、synchronizeまで成功することを検証する。

期待するruntime identity:

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

最終修正後のcanonical Vast proofでは次を記録した。

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

proof marker:

```text
JPA_CF_CANONICAL_VERIFY rc=0
```

HF publicationとGitHub artifact upload後、instance `49592872` はdestroyされた。

これらの値はhistorical evidenceであり、将来runの固定設定ではない。今後のauthoritative runは毎回新しいdigest/provider metadataを記録する。

---

## 17. Hugging Faceの責務

GHCRとHugging Faceは競合するstorageではなく役割が異なる。

```text
GHCR
  executable immutable environment

HF Bucket
  transferable files / cache / runtime identity / experiment evidence
```

GPU rootfs全体をHF Bucketで運ばない。

一方で、実験evidenceやcross-providerで再利用したいartifactをVast local diskだけに残さない。

---

## 18. HF transport environmentの分離

Hugging Face Bucket CLI用environmentはASR runtimeと分離し、次に配置する。

```text
/opt/jpacf/tools/hf-bucket
```

`scripts/hf/hf-identity.sh` の `hf_bucket_cli` がtransportを担当する。

canonical container内では、すでにimage build時にmaterialize済みのtransport environmentを使い、`--no-sync` で実行する。

これによりevidence uploadの途中でuvがtransport dependencyを再解決・再同期し、ASR runtimeのlocked stateへ影響することを防ぐ。

container外では必要に応じてisolated transport projectのみ同期する。

---

## 19. HF Bucket namespace

project bucket:

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket
```

namespaceは責務ごとに分ける。

```text
runtime/sha-<git-sha>/
  verified runtime identity / provider proof

workspace-cache/<deterministic-key>/
  artifacts/
  generated/

runs/<run-id>/
  immutable experiment evidence
```

### runtime identity

GPU verification成功後に `runtime-image.json` を作成する。

主な内容:

```text
source_git_sha
platform
CUDA major
NeMo version
torch version
runtime_base_reference
runtime digest
full image reference
gpu_verified
execution_contract
state_mount
Vast provider metadata（Vast proof時）
```

そして次へsyncする。

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/runtime/sha-<source-sha>
```

canonical proof例:

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/runtime/sha-5c9c79856093eded02f67dfe148f56e99f607a9b
```

これによりGit source SHAから「実際にGPUで検証を通過したruntime digest」をprovider非依存で参照できる。

---

## 20. cross-provider workspace cache

すべてのlocal cacheをHFへ同期するのではなく、再生成コストが高くprovider間で価値のあるものだけを転送する。

転送対象:

```text
artifacts/
generated/
```

provider-localに残すもの:

```text
hf/
uv/
xdg/
torch/
```

local package/download cacheは巨大になりやすく、lockから再生成可能なのでcross-provider transferの費用対効果が低い。

### deterministic key

keyはrepositoryのinput/lock/materialization contractから生成する。

```bash
KEY="$(bash scripts/container/cache-key.sh)"
```

remote path:

```text
hf://buckets/<bucket>/workspace-cache/<KEY>
```

既存keyはimmutableとして扱い、上書きしない。

inputが変化したらnew keyを生成する。

### plan/apply

`scripts/hf/hf-sync-workspace-cache.sh` はpush時にHF syncのplan/applyを利用する。

```text
artifacts upload planを生成
generated upload planを生成
planをapply
```

既存keyが存在する場合はpublicationを拒否する。

これによりworkspace cacheを「mutable共有folder」ではなく、content-addressedに近いresearch materialとして扱う。

---

## 21. Vast VolumeとHF Bucketの違い

### Vast Volume / self-hosted local state

利点:

```text
同一hostで高速
大量の一時cacheを保持可能
再起動時のmaterializationを削減
```

欠点:

```text
provider/host placementに依存
別Vast hostへ自動で追従するとは限らない
単体ではauthoritative evidenceにならない
```

### HF Bucket

利点:

```text
cross-provider
明示的namespace
runtime identityやimmutable evidenceに向く
新しいGPU hostでartifactを復元可能
```

欠点:

```text
local diskよりnetwork latencyがある
全package cacheをlive filesystemのように置く用途には向かない
```

したがって最終方針は次。

```text
provider-local state = speed
HF Bucket            = portability + evidence
```

---

## 22. End-to-end canonical flow

```text
1. pinned NVIDIA base digestを読む
        |
2. dependency base keyを計算
        |
3. base-<key>をreuse、なければ一度だけbuild
        |
4. dependency baseをdigestへ解決
        |
5. thin source runtimeをbuild
        |
6. BuildxからGHCRへdirect push
        |
7. exact runtime digestを解決
        |
8. GPU targetを選択
        |
9. exact digestをreal NVIDIA GPUで起動
        |
10. CUDA tensor compute proof
        |
11. runtime identityを記録
        |
12. HF Bucketへidentity/evidenceをsync
        |
13. Vast使用時はpaid instanceをdestroy
        |
14. verified digestでexperiment実行
```

experiment側では必要に応じて以下を追加する。

```text
HF workspace-cache restore
 -> missing artifacts/generatedをprepare
 -> E00-E06 / E07a実行
 -> immutable run evidence publication
```

---

## 23. GitHub Actionsで必要なsecret名

orchestrationではrepository secretとして次を使用する。

```text
VAST_API_KEY
HF_TOKEN
```

GHCRにはworkflow tokenをpackage permission付きで使用する。

secret valueをrepository file、Docker layer、build metadata、diagnostic artifactへ保存してはならない。

---

## 24. 代表的な運用コマンド

### promoted runtimeをdigestへ解決

```bash
export JPA_CF_IMAGE="$(
  bash scripts/container/resolve-image.sh \
    ghcr.io/yokane/jpacf-yomi-tdt-runtime:main
)"
```

### local / self-hosted NVIDIA runtimeで検証

```bash
bash scripts/container/run.sh \
  python /opt/jpacf/scripts/container/verify_runtime.py --require-gpu
```

### workspace cache key生成

```bash
KEY="$(bash scripts/container/cache-key.sh)"
```

### workspace cache restore

```bash
bash scripts/container/run.sh \
  bash scripts/hf/hf-sync-workspace-cache.sh pull "$KEY"
```

### new immutable workspace cache publish

```bash
bash scripts/container/run.sh \
  bash scripts/hf/hf-sync-workspace-cache.sh push "$KEY"
```

### 有料Vast canonical proof

GitHub Actionsのmanual workflowを使用する。

```text
ghcr-runtime-vast-fallback
```

---

## 25. Troubleshooting matrix

| 症状 | 主な原因layer | 最初に確認するもの |
|---|---|---|
| Docker stage完了後にrunnerがexport中に終了 | hosted Docker image-store I/O | `docker-container` + `--push`、`--load`を使っていないか |
| source変更だけでdependency baseが再build | base key scope | `build_runtime_image.sh` の `base_key` inputs |
| state mount後にproject Pythonが消える | interpreterがmutable state配下 | `/opt/jpacf/.uv-python` invariant |
| self-hosted staged runでpermission denied | root-owned persistent state | shared wrapperとhost UID/GID |
| Vast logでshell command全体がpath扱い | provider argv quoting | `vast_verify.sh` absolute path |
| Vast container終了後もworkflowが待つ | fail-fast不足 | `actual_status == exited` とproof marker判定 |
| Vast課金が止まらない | cleanup path | persisted instance ID / `always()` destroy |
| HF upload時にruntime dependencyが変わる | transport env混在 | isolated `hf_bucket_cli` / container `--no-sync` |
| 新GPU hostにartifactがない | local volumeをportableと誤認 | HF `workspace-cache/<key>` restore |
| tagとevidenceのruntimeが一致しない | mutable tagをidentityに使用 | execution前の `@sha256:` resolve |

---

## 26. 今後も維持すべきarchitecture invariant

1. managed Pythonを `/workspace/state` 配下へ置かない。
2. persistent provider stateを `/opt/jpacf` にmountしない。
3. hosted canonical buildで巨大imageへ `--load` を使わない。
4. build成功だけでruntimeをpromoteせず、exact digestをreal GPUで検証する。
5. experiment identityにmutable tagを使用しない。
6. ad-hoc root containerでself-hosted persistent stateを作らない。
7. Vast `--args` に複雑なquoted shell programを渡さない。
8. Vast destroyを前step成功に依存させない。
9. research run中にHF transport dependencyをASR runtimeへ再同期しない。
10. deterministic HF workspace-cache keyを上書きしない。
11. Vast Volumeをcross-provider source of truthとみなさない。
12. hosted runner上でpush直後の巨大imageをmanifest確認目的だけでpull-backしない。

---

## 27. Context7で確認したupstream documentation

今回の設計はContext7経由で現行upstream documentationを再確認した。

### Docker

- <https://docs.docker.com/build/builders/drivers/>
- <https://docs.docker.com/build/exporters/image-registry/>
- <https://docs.docker.com/build/cache/backends/registry/>
- <https://docs.docker.com/build/cache/optimize/>

確認した重要点:

```text
docker-container driverはbuild resultをlocal image storeへ自動loadしない
--pushでregistry exporterへ直接出力できる
--loadはnon-default builderのresultをlocal image storeへloadする明示的経路
cache exporter/importerはfinal immutable image identityと分離可能
RUN --mount=type=cacheはperformance optimizationでありruntime correctnessに依存させない
```

### Hugging Face Hub

- <https://huggingface.co/docs/huggingface_hub/guides/cli>
- <https://huggingface.co/docs/huggingface_hub/guides/buckets>

確認した重要点:

```text
hf buckets syncはlocal -> bucket / bucket -> localのdirectory syncを扱える
--plan / --apply によるtwo-phase syncが可能
authenticationはsync対象dataとは分離して与える
通常syncでは必要なoperationだけが転送対象になる
```

---

## 28. 関連ファイル

### Docker / runtime

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

### Vast

```text
.github/workflows/ghcr-runtime-vast-fallback.yml
scripts/providers/vast/build_search_query.py
scripts/providers/vast/rank_offers.py
```

### Hugging Face

```text
configs/hf-storage.json
scripts/hf/hf-identity.sh
scripts/hf/hf-sync-workspace-cache.sh
tools/hf-bucket/
```

### 利用手順

```text
docs/portable-gpu-runtime.md
```

### regression contract

```text
tests/test_container_contract.py
tests/test_runtime_managed_python_contract.py
tests/test_hf_bucket_storage.py
tests/test_self_hosted_cache_contract.py
```

このarchitectureを変更する場合、implementationだけでなくcontract testも同時に更新し、重要invariantをmachine-checkableな状態に維持する。
