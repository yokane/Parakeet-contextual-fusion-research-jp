# E00–E06 研究Artifact・実行環境・処理手順

この文書は `nvidia/parakeet-tdt_ctc-0.6b-ja` を基盤とする E00–E06 研究を、毎回CUDA/NeMo環境や研究資材を作り直さず、再現可能かつ低コストに反復するための運用契約です。

機械可読な対応表は [`configs/research/e00-e06-artifacts.yaml`](../configs/research/e00-e06-artifacts.yaml) にあります。

GPUを借りる前には必ずArtifact readinessを確認します。

```bash
uv run --locked --no-sync python scripts/research/check_phase_artifacts.py E04 \
  --state-root /workspace/state
```

## 1. 基本方針

処理を3つのplaneへ分離します。

```text
GitHub-hosted CPU                          Vast GPU
-----------------                          --------
benchmark/audio materialization            model/tokenizer依存処理
KenLM lmplz/build_binary                    E00/E01 decode
E05 phone-head学習・rerank                  E02 tokenizer encode / pack / decode
static validation                           E03 GPU phrase biasing
                                            E04 hybrid CTC rerank
                                            E05 encoder extraction
                                            E06 in-beam integration
             \                              /
              \                            /
               Hugging Face Bucket
               immutable delta snapshots
               + append-only run evidence
```

原則は次の通りです。

1. **CPUで可能な処理はGitHub-hosted runnerで行う。**
2. **0.6B ASR model、CUDA、NeMo内部状態が必要な処理だけVastへ送る。**
3. **Container registryにはsoftware environmentだけを置く。**
4. **audio、trained LM、encoder tensor、phone head、結果はHF Bucketへ置く。**
5. **GitHub Actions cacheへDocker layerや研究データを入れない。**
6. **Vastを確保する前に必要Artifactの存在を確認する。**
7. **既に完了したimmutable snapshotが存在するtaskは再計算しない。**

## 2. 固定identity

Public benchmark/model:

```text
Dataset: saeeew/JP-HomophoneBench
Model:   saeeew/J-PACF-YOMI-tdt
Bucket:  saeeew/J-PACF-YOMI-tdt-bucket
```

実験で使うexact revisionは `locks/hf-revisions.lock.json` を唯一のsource of truthとします。

## 3. research key

同じ研究資材を再利用する単位を `research_key` とします。

```bash
uv run --locked --no-sync python scripts/research/research_key.py
```

形式:

```text
v1-bench-<12hex>-model-<12hex>-ng6
```

workflow inputの `research_key` を空欄にすればlocked revisionから同じ値が導出されます。

## 4. HF Bucketは「mutable workspace」ではなくimmutable delta snapshotにする

既存の `workspace-cache/<key>` immutable契約を壊さないため、各taskは研究workspace全体を上書きしません。

Remote layout:

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/
└── workspace-cache/
    └── e00-e06/
        └── <research-key>/
            ├── common/
            ├── e02-encode/
            ├── e02-estimate/
            ├── e02-pack/
            ├── phase-e00/
            ├── phase-e01/
            ├── phase-e02/
            ├── phase-e03/
            ├── phase-e04/
            ├── e05-extract/
            ├── e05-phone/
            └── phase-e06/
```

各directoryは**一度作成したら上書き禁止**です。

各taskは次の順で動きます。

```text
required snapshot(s)
        |
        | overlay restore
        v
/workspace/state
        |
        | current task only
        v
new artifacts
        |
        | selected paths only
        v
new immutable delta snapshot
```

つまり `common/` のaudioをE02/E03/E04 snapshotへ複製して保存しません。consumer側で必要なdeltaをlocal stateへoverlayします。

snapshot操作は次で統一します。

```bash
bash scripts/hf/hf-research-snapshot.sh plan E04
bash scripts/hf/hf-research-snapshot.sh exists "$RESEARCH_KEY" E04
bash scripts/hf/hf-research-snapshot.sh pull   "$RESEARCH_KEY" E04 /workspace/state
bash scripts/hf/hf-research-snapshot.sh push   "$RESEARCH_KEY" E04 /workspace/state
```

各snapshotには次も保存されます。

```text
.jpacf-snapshots/<stage>.json
```

ここにはsource Git SHA、task/stage、file size、SHA-256 inventoryを記録します。

## 5. snapshot lineage

| Task | Executor | 読み込むsnapshot | 新規snapshot |
|---|---|---|---|
| `common` | GitHub-hosted | - | `common` |
| `e02-encode` | Vast | `common` | `e02-encode` |
| `e02-estimate` | GitHub-hosted | `e02-encode` | `e02-estimate` |
| `e02-pack` | Vast | `e02-encode`, `e02-estimate` | `e02-pack` |
| `E00` | Vast | `common` | `phase-e00` |
| `E01` | Vast | `common` | `phase-e01` |
| `E02` | Vast | `common`, `e02-pack` | `phase-e02` |
| `E03` | Vast | `common`, `e02-pack` | `phase-e03` |
| `E04` | Vast | `common`, `e02-pack` | `phase-e04` |
| `e05-extract` | Vast | `common` | `e05-extract` |
| `e05-phone` | GitHub-hosted | `common`, `phase-e04`, `e05-extract` | `e05-phone` |
| `E06` | Vast | `common`, `e02-pack`, `e05-extract`, `e05-phone` | `phase-e06` |

同一 `research_key + output stage` がすでに存在する場合、CPU workflowは計算をskipし、Vast workflowは**GPU instanceを作成する前に終了**します。

## 6. Container image設計

### 6.1 重いGPU runtimeは1つだけ

authoritative parent:

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime@sha256:<digest>
```

このparentが保持するもの:

- Linux/amd64
- CUDA 13
- Python 3.12.3
- uv 0.12.1
- torch 2.12.0+cu132
- NeMo 3.0.0
- project runtime
- HF Bucket transport environment

phaseごとにこれらを再installしません。

### 6.2 thin GPU phase image

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e00-<git-sha>
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e01-<git-sha>
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e02-<git-sha>
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e03-<git-sha>
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e04-<git-sha>
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e05-<git-sha>
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e06-<git-sha>
```

E00/E01/E03/E04/E05/E06は `docker/phases/Dockerfile` のnamed targetです。

E02だけは `docker/research/Dockerfile.e02` を使い、KenLM runtime binariesを追加します。ただしKenLM compiler/source treeは含めません。

### 6.3 独立CPU tool image

```text
ghcr.io/yokane/jpacf-yomi-tdt-tools:kenlm-<revision-short>
ghcr.io/yokane/jpacf-yomi-tdt-tools:phone-e05-<git-sha>
```

KenLM image:

```text
Debian slim
+ lmplz
+ build_binary
+ minimum runtime libraries
```

E05 phone image:

```text
Python 3.12.3 slim
+ CPU PyTorch
+ PhoneCTCHead scripts
```

CUDA/NeMo/model checkpointは含めません。

## 7. Common Artifact — GitHub-hosted CPU

Workflow:

```text
research-artifacts-cpu
  task=common
```

Producer:

```text
scripts/research/prepare_common_artifacts.sh
```

Output snapshot `common/`:

```text
generated/eval/bench_index.jsonl
generated/eval/nemo_eval.jsonl
generated/eval/audio/*
generated/eval/eval_provenance.json
generated/eval/audio_coverage.json
generated/eval/context_phrases.txt
generated/eval/lm_corpus.txt
```

`nemo_eval.jsonl` のabsolute audio pathはprovider間で変わるため、Vast restore後に `scripts/research/rebase_eval_manifest.py` が `/workspace/state/generated/eval/audio/...` へrebaseします。

## 8. E00 — TDT greedy baseline

Executor: **Vast GPU**

Input snapshot:

```text
common
```

Required:

```text
generated/eval/nemo_eval.jsonl
generated/eval/audio/*
```

Output snapshot `phase-e00/`:

```text
results/E00_tdt_greedy.jsonl
```

目的はbeam/LM/context biasなしの基準性能を固定することです。

## 9. E01 — TDT MAES/beam

Executor: **Vast GPU**

Input:

```text
common
```

Output `phase-e01/`:

```text
results/E01_tdt_beam.jsonl
```

beam size等のsearch parameterはrun evidenceに残します。

## 10. E02 — KenLM / NGPU-LM

E02は1つのcontainerで全部処理しません。dependency boundaryで3段階に分離します。

```text
lm_corpus.txt
     |
     | e02-encode / Vast
     | exact NeMo + locked Parakeet tokenizer
     v
lm_corpus.encoded.txt
encoding-metadata.json
     |
     | e02-estimate / GitHub-hosted CPU
     | pinned KenLM lmplz/build_binary
     v
ja-6gram.arpa
ja-6gram.binary
estimation-metadata.json
     |
     | e02-pack / Vast
     | exact NeMo NGramGPULanguageModel
     v
ja-6gram.nemo
package-metadata.json
     |
     | E02 / Vast
     v
E02_ngpulm.jsonl
```

### 10.1 e02-encode — Vast

Input snapshot:

```text
common
```

Output `e02-encode/`:

```text
artifacts/lm/lm_corpus.encoded.txt
artifacts/lm/encoding-metadata.json
```

NeMoのsubword KenLM pathはtoken IDをUnicode symbolへ変換するため、ここだけlocked tokenizer/NeMo環境が必要です。

ASR `.nemo` checkpointはVast local scratchへmaterializeしますが、snapshotへはpublishしません。

### 10.2 e02-estimate — GitHub-hosted CPU

Input:

```text
e02-encode
```

Tool image:

```text
jpacf-yomi-tdt-tools:kenlm-<revision>
```

Pinned KenLM revision:

```text
4cb443e60b7bf2c0ddf3c745378f76cb59e254e5
```

Output `e02-estimate/`:

```text
artifacts/lm/ja-6gram.arpa
artifacts/lm/ja-6gram.binary
artifacts/lm/estimation-metadata.json
```

### 10.3 e02-pack — Vast

Inputs:

```text
e02-encode
e02-estimate
```

Output `e02-pack/`:

```text
artifacts/lm/ja-6gram.nemo
artifacts/lm/package-metadata.json
```

### 10.4 E02 decode — Vast

Inputs:

```text
common
e02-pack
```

Output `phase-e02/`:

```text
results/E02_ngpulm.jsonl
```

## 11. E03 — GPU phrase/context biasing

Executor: **Vast GPU**

Inputs:

```text
common
e02-pack
```

Required:

```text
generated/eval/nemo_eval.jsonl
generated/eval/context_phrases.txt
artifacts/lm/ja-6gram.nemo
```

Output `phase-e03/`:

```text
results/E03_gpu_pb.jsonl
```

追加のheavy dependency imageは不要です。NeMo word boosting/boosting-tree機能はcanonical runtimeを利用します。

## 12. E04 — local hybrid CTC N-best rerank

Executor: **Vast GPU**

Inputs:

```text
common
e02-pack
```

Outputs `phase-e04/`:

```text
results/E04_nbest.jsonl
results/E04_ctc_rerank.jsonl
```

locked modelのCTC branchを再実行するためGPU taskです。

## 13. E05 — frozen encoder phoneme CTC rerank

E05はcanonical pathをGPU/CPUに分割します。

### 13.1 encoder extraction — Vast

Task:

```text
e05-extract
```

Input:

```text
common
```

Output `e05-extract/`:

```text
artifacts/encoder/*.pt
```

0.6B Parakeet encoder forwardだけをVastへ残します。

### 13.2 phone-head train + rerank — GitHub-hosted CPU

Task:

```text
e05-phone
```

Inputs:

```text
common
phase-e04
e05-extract
```

Outputs `e05-phone/`:

```text
artifacts/phone_vocab.json
artifacts/phone_head.pt
generated/phone_train.jsonl
results/E04_phone_ready.jsonl
results/E05_phone_rerank.jsonl
```

small projectionの学習とCTC scoringはCPUで実行できるため、Vastを使用しません。

`run-phase.sh E05` はmonolithic debugging routeとして残しますが、canonical research workflowでは使用しません。

## 14. E06 — version-isolated in-beam fusion

Executor: **Vast GPU**

Inputs:

```text
common
e02-pack
e05-extract
e05-phone
```

Required:

```text
generated/eval/nemo_eval.jsonl
generated/eval/context_phrases.txt
artifacts/lm/ja-6gram.nemo
artifacts/encoder/*.pt
artifacts/phone_head.pt
artifacts/phone_vocab.json
E06_DRIVER=<pinned NeMo-3.0.0-specific driver>
```

Output `phase-e06/`:

```text
results/E06_inbeam.jsonl
```

`patches/README.md` のpromotion gateを満たすdriverが作られるまでは、E06は意図的に明示driver指定を要求します。新しいCUDA baseは作らず、driver/patchだけを `phase-e06` へthin overlayする方針です。

## 15. 推奨実験順序

```text
common (GitHub-hosted)
  |
  +------> E00 (Vast)
  +------> E01 (Vast)
  |
  +------> e02-encode (Vast)
             |
             v
          e02-estimate (GitHub-hosted)
             |
             v
          e02-pack (Vast)
             |
             +------> E02 (Vast)
             +------> E03 (Vast)
             +------> E04 (Vast)
                        |
                        +--> e05-extract (Vast; commonから独立実行も可)
                        |         |
                        +---------+--> e05-phone (GitHub-hosted)
                                         |
                                      E05 result
                                         |
                                    promotion gate
                                         |
                                         v
                                       E06 (Vast)
```

Canonical manual sequence:

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
research-phase-vast:    task=E06   # promotion gate後のみ
```

E00/E01とE02 preparationは独立しているため並列研究が可能です。

## 16. Buildx / registry方針

`research-images` workflowはGitHub-hosted CPU上でsoftware imageをbuildします。

```text
Buildx driver: docker-container
output:        direct registry --push
--load:        使用しない
Docker GHA cache: 使用しない
```

理由:

- huge CUDA/NeMo parentはregistryにすでに存在する;
- phase overlayは小さい;
- `--load`するとmulti-GB parentをhost Docker image storeへmaterializeしてしまう;
- immutable tagが存在すればbuild自体をskipできる;
- Actions cacheをDocker layerで圧迫しない。

Dockerのregistry exporter/cacheはfinal imageとcacheを分離できますが、このthin phase群では大量のregistry cache refを増やすより、immutable software tagの再利用を優先します。

## 17. image build cache key

KenLM tools:

```text
kenlm-<KENLM_REVISION short>
```

E05 CPU tool:

```text
phone-e05-<SOURCE_SHA>
```

GPU phase:

```text
phase-eXX-<SOURCE_SHA>
```

build scriptは `docker buildx imagetools inspect` でtag存在を先に確認します。存在すればrebuildしません。

`dist/research-images.json` に実際のregistry/tag/digest/runtime parentを記録します。

## 18. GHCR → public Docker Hub fallback

Primary registryはGHCRです。

Optional secrets:

```text
DOCKERHUB_ACCESS_TOKEN
DOCKERHUB_REPOSITORY
```

`DOCKERHUB_REPOSITORY` 例:

```text
namespace/jpacf-yomi-tdt-research
```

publication policy:

1. GHCR immutable tagを検索;
2. あれば再利用;
3. なければGHCRへdirect push;
4. GHCR pushが失敗した場合のみDocker Hubへ同じsuffixでbuild/push;
5. GHCR成功時はDocker Hubへ二重mirrorしない。

consumer側もGHCR image取得に失敗し、`DOCKERHUB_REPOSITORY` が設定されている場合はpublic `docker.io/<repo>:<same-tag>` を試します。

これによりGHCR quota/availability問題が起きてもartifact contractを変えずに研究を継続できます。

## 19. Storage rule

### Container registryに置いてよいもの

```text
software layers
KenLM executable
phase dispatcher
CPU phone-head runtime
small orchestration scripts
OCI labels/manifests
```

### Container registryへ入れてはいけないもの

```text
evaluation audio
ASR modelを研究artifactとして複製したもの
trained ARPA/binary/NGPU-LM
encoder *.pt
phone_head.pt
result JSONL/Parquet
HF evidence
```

### HF Bucket

保存対象:

```text
common audio/manifests
encoded corpus
ARPA/binary/.nemo LM
e02 metadata
encoder states
phone artifacts
phase results
append-only runs/<run-id> evidence
```

ただし各stageはdeltaだけを保存し、既存snapshotを上書きしません。

### GitHub Actions cache

mise/uv等のsmall tool cacheに限定します。

```text
禁止: type=gha,mode=max をresearch image buildへ追加
禁止: audio/LM/encoder tensorsをActions cacheへ保存
```

### GitHub workflow artifact

14日程度のshort-lived control evidenceだけにします。

```text
research-images.json
Vast selected offer
Vast create response
instance status
short log
```

## 20. Artifact readiness

```bash
STATE=/workspace/state
uv run --locked --no-sync python scripts/research/check_phase_artifacts.py E00 --state-root "$STATE"
uv run --locked --no-sync python scripts/research/check_phase_artifacts.py E02 --state-root "$STATE"
uv run --locked --no-sync python scripts/research/check_phase_artifacts.py E04 --state-root "$STATE"
uv run --locked --no-sync python scripts/research/check_phase_artifacts.py E05 --state-root "$STATE"
uv run --locked --no-sync python scripts/research/check_phase_artifacts.py E06 --state-root "$STATE"
```

readiness failureはGPU experiment failureではありません。producer snapshotを先に作成してください。

## 21. Vast課金を避けるgate

`research-phase-vast.yml` は次の順で処理します。

```text
research_key resolve
    |
    v
HF output snapshot exists?
    |
    +-- yes --> success / GPU allocationなし
    |
    +-- no --> image resolve
                  |
                  v
             offer search
                  |
                  v
             Vast create
                  |
                  v
              execute
                  |
                  v
       immutable delta publish
                  |
                  v
        destroy instance(always)
```

Vast instance destroyは先行step成功に依存させません。

## 22. 再現性metadata

少なくとも次を保持します。

- benchmark repo + full revision
- base model repo + full revision
- source Git SHA
- exact phase image digest
- runtime parent digest
- KenLM revision
- N-gram order
- source corpus SHA-256
- encoded corpus SHA-256
- ARPA SHA-256
- binary SHA-256
- NGPU-LM SHA-256
- NeMo 3.0.0
- torch 2.12.0+cu132
- E05 phone-head hyperparameters
- encoder feature model revision
- Vast offer/GPU/cost
- research key
- snapshot stage manifest
- immutable `runs/<run-id>` evidence

## 23. 失敗時の扱い

### Snapshotが足りない

producer taskへ戻ります。GPUを再度借りないでください。

### 同じsnapshotが既に存在する

正常なcache hitです。上書きせず再利用します。

### GHCR push/pull failure

Docker Hub fallbackが設定されていれば同tagを使用します。

### Vast task failure

`dist/runtime/` とprovider inventoryを確認します。instanceは `if: always()` cleanupでdestroyします。

### Audio pathが旧providerを指す

`rebase_eval_manifest.py` を使います。snapshot内のaudio実体そのものは変更しません。

### E02 LMがdriftした

encoding/estimation/package metadataのSHA-256 chainで発生段階を特定します。

## 24. E06 promotion gate

E06はE04/E05でreproducibleなN-best gainが得られた後に進めます。

その時点で:

```text
patches/nemo-<short-sha>/
├── README.md
├── inbeam_driver.py
└── nemo.patch
```

を作り、NeMo commitとdriverを固定します。

新しいCUDA runtimeを作るのではなく、既存 `phase-e06` にdriver/patchだけをthin overlayするのが原則です。

## 25. Upstream design references

この設計ではcurrent upstream contractを基準にしています。

- Docker Buildx: `docker-container` builder + registry/image exporter + direct `--push`
- Docker cache: registry cacheはfinal imageとは分離可能。複数targetではcache refを分ける必要がある
- KenLM: CMake build、`lmplz`、`build_binary`
- NVIDIA NeMo Speech: TDT/RNNT beam + N-gram LM、`malsd_batch`、GPU boosting tree
- Hugging Face Bucket: local↔bucket syncとplan/apply

upstream exampleは設計確認用であり、実際の研究identityはrepository lock・digest・snapshot manifestを優先します。
