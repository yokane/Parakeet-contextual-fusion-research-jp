# E00–E06 研究Artifact・実行環境・処理手順

この文書は `nvidia/parakeet-tdt_ctc-0.6b-ja` を基盤とする E00–E06 研究を、**CPUで可能な処理はGitHub-hosted、GPUが必要な処理だけVast**へ分離し、研究Artifact・container・HF Bucketを再利用しながら反復するための運用契約です。

機械可読な契約は [`configs/research/e00-e06-artifacts.yaml`](../configs/research/e00-e06-artifacts.yaml)、content-addressed snapshotの詳細は [`research-snapshot-fingerprints.md`](research-snapshot-fingerprints.md) を参照してください。

## 1. 研究plane

```text
Local / Dev Container / Codex Cloud
  編集・lint・CPU unit test・workflow/Dockerfile開発
                    |
                    v
GitHub-hosted CPU                         Vast GPU
-----------------                         --------
common benchmark/audio                    E00/E01 decode
KenLM lmplz/build_binary                   E02 tokenizer encode / NeMo pack / decode
E05 phone-head train/rerank                E03 GPU phrase biasing
static/reproducibility checks              E04 hybrid CTC rerank
thin image Buildx                          E05 encoder extraction
                                           E06 in-beam integration
                    \                     /
                     \                   /
                      Hugging Face Bucket
                      immutable delta snapshots
                      + append-only run evidence
```

原則:

1. CPUで完結する処理はGitHub-hosted runnerで実行する。
2. ASR model forward、CUDA、NeMo tokenizer/decoder内部状態が必要な処理だけVastへ送る。
3. GHCR/Docker Hubにはsoftware environmentだけを置く。
4. audio、LM、encoder tensor、phone head、結果はHF Bucketへ置く。
5. GitHub Actions cacheへDocker layerや研究Artifactを保存しない。
6. GPU allocation前にsnapshot existenceとsource-matched imageを解決する。
7. immutable snapshotが存在するtaskは再計算しない。
8. mutable `*-current` imageはcanonical research executionに使わない。

## 2. 固定identity

```text
Dataset: saeeew/JP-HomophoneBench
Model:   saeeew/J-PACF-YOMI-tdt
Bucket:  saeeew/J-PACF-YOMI-tdt-bucket
```

exact revisionは `locks/hf-revisions.lock.json` をsource of truthとします。

## 3. research key と stage fingerprint

データ/model側のidentity:

```bash
RESEARCH_KEY="$(uv run --locked --no-sync python scripts/research/research_key.py)"
```

形式:

```text
v1-bench-<benchmark12>-model-<model12>-ng6
```

実装側のidentityはtaskごとの `stage-fingerprint` です。

```bash
uv run --locked --no-sync python scripts/research/stage_fingerprints.py \
  --task E04 --field fingerprint
```

fingerprintは以下をSHA-256へ正規化します。

- task固有の実装file/directory
- task固定parameter
- upstream stage fingerprint
- E06のように必要なexternal implementation identity

このためE05だけを修正した場合、common/KenLM/E00–E04は再利用し、`e05-phone`とそのconsumer E06だけをinvalidateできます。

## 4. HF Bucket immutable delta snapshot

Canonical remote layout:

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/
└── workspace-cache/
    └── e00-e06/
        └── <research-key>/
            ├── common/<fingerprint>/
            ├── e02-encode/<fingerprint>/
            ├── e02-estimate/<fingerprint>/
            ├── e02-pack/<fingerprint>/
            ├── phase-e00/<fingerprint>/
            ├── phase-e01/<fingerprint>/
            ├── phase-e02/<fingerprint>/
            ├── phase-e03/<fingerprint>/
            ├── phase-e04/<fingerprint>/
            ├── e05-extract/<fingerprint>/
            ├── e05-phone/<fingerprint>/
            └── phase-e06/<fingerprint>/
```

snapshotは**一度作成したら上書きしません**。継承Artifactを複製せず、各taskが新規生成したdeltaだけをpublishします。consumerは必要なsnapshotを `/workspace/state` へ `--no-delete` overlay restoreします。

操作:

```bash
bash scripts/hf/hf-research-snapshot.sh plan   E04
bash scripts/hf/hf-research-snapshot.sh remote "$RESEARCH_KEY" E04
bash scripts/hf/hf-research-snapshot.sh exists "$RESEARCH_KEY" E04
bash scripts/hf/hf-research-snapshot.sh pull   "$RESEARCH_KEY" E04 /workspace/state
bash scripts/hf/hf-research-snapshot.sh push   "$RESEARCH_KEY" E04 /workspace/state
```

各snapshotには次を保存します。

```text
.jpacf-snapshots/<stage>-<fingerprint-prefix>.json
```

manifestには `research_key`, `task`, `stage`, `fingerprint`, `input_refs`, `output_ref`, source Git SHA、file size/SHA-256 inventoryを記録します。

## 5. 全task lineage

| Task | Executor | Input snapshot | Output snapshot |
|---|---|---|---|
| `common` | GitHub-hosted | - | `common/<fp>` |
| `e02-encode` | Vast | `common/<fp>` | `e02-encode/<fp>` |
| `e02-estimate` | GitHub-hosted | `e02-encode/<fp>` | `e02-estimate/<fp>` |
| `e02-pack` | Vast | `e02-encode/<fp>`, `e02-estimate/<fp>` | `e02-pack/<fp>` |
| `E00` | Vast | `common/<fp>` | `phase-e00/<fp>` |
| `E01` | Vast | `common/<fp>` | `phase-e01/<fp>` |
| `E02` | Vast | `common/<fp>`, `e02-pack/<fp>` | `phase-e02/<fp>` |
| `E03` | Vast | `common/<fp>`, `e02-pack/<fp>` | `phase-e03/<fp>` |
| `E04` | Vast | `common/<fp>`, `e02-pack/<fp>` | `phase-e04/<fp>` |
| `e05-extract` | Vast | `common/<fp>` | `e05-extract/<fp>` |
| `e05-phone` | GitHub-hosted | `common/<fp>`, `phase-e04/<fp>`, `e05-extract/<fp>` | `e05-phone/<fp>` |
| `E06` | Vast | `common/<fp>`, `e02-pack/<fp>`, `e05-extract/<fp>`, `e05-phone/<fp>` | `phase-e06/<fp>` |

## 6. Container image設計

重いCUDA/NeMo rootfsは1つのauthoritative runtimeで共有します。

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime@sha256:<digest>
```

ここにLinux/amd64、CUDA 13、Python 3.12.3、uv 0.12.1、torch 2.12.0+cu132、NeMo 3.0.0を固定します。

phase imageはsource Git SHAをtagへ含めます。

```text
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e00-<git-sha>
...
ghcr.io/yokane/jpacf-yomi-tdt-runtime:phase-e06-<git-sha>
```

E00/E01/E03/E04/E05/E06は `docker/phases/Dockerfile` のthin named targetです。E02だけはKenLM runtime binariesを追加する `docker/research/Dockerfile.e02` を使います。

CPU tool image:

```text
ghcr.io/yokane/jpacf-yomi-tdt-tools:kenlm-4cb443e60b7b
ghcr.io/yokane/jpacf-yomi-tdt-tools:phone-e05-<git-sha>
```

KenLM compiler/source、audio、model checkpoint、LM、encoder tensor、resultsはimage layerへ入れません。

Buildは `docker-container` Buildx + direct `--push` を使い、`--load`とDocker `type=gha` cacheは使いません。既存immutable tagがremoteにあればbuild自体をskipします。

GHCR push/pullが利用できない場合のみ、repository secrets `DOCKERHUB_ACCESS_TOKEN` と `DOCKERHUB_REPOSITORY` を使って同一tagのpublic Docker Hub imageへfallbackします。GHCR成功時の二重mirrorは行いません。

## 7. Common Artifact — GitHub-hosted CPU

Workflow:

```text
research-artifacts-cpu / task=common
```

Output:

```text
generated/eval/bench_index.jsonl
generated/eval/nemo_eval.jsonl
generated/eval/audio/*
generated/eval/eval_provenance.json
generated/eval/audio_coverage.json
generated/eval/context_phrases.txt
generated/eval/lm_corpus.txt
```

Vast restore後、`nemo_eval.jsonl` のabsolute audio pathは `scripts/research/rebase_eval_manifest.py` で `/workspace/state/generated/eval/audio/...` へrebaseします。

## 8. E00 — TDT greedy baseline

Executor: **Vast GPU**

Input:

```text
common
```

Required:

```text
generated/eval/nemo_eval.jsonl
generated/eval/audio/*
```

Output:

```text
results/E00_tdt_greedy.jsonl
```

beam/LM/context biasを使用しない基準性能です。

## 9. E01 — TDT beam

Executor: **Vast GPU**

Input: `common`

Output:

```text
results/E01_tdt_beam.jsonl
```

E00との差からsearchだけによる改善を評価します。

## 10. E02 — KenLM / NGPU-LM

E02はdependency boundaryで3段階に分離します。

```text
lm_corpus.txt
  |
  | e02-encode / Vast
  | locked Parakeet tokenizer + NeMo
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
  | NeMo NGramGPULanguageModel packaging
  v
ja-6gram.nemo
package-metadata.json
  |
  | E02 / Vast
  v
E02_ngpulm.jsonl
```

Pinned KenLM revision:

```text
4cb443e60b7bf2c0ddf3c745378f76cb59e254e5
```

### e02-encode

Vastでlocked tokenizerを使い、subword IDをKenLM用Unicode token corpusへ変換します。

Output:

```text
artifacts/lm/lm_corpus.encoded.txt
artifacts/lm/encoding-metadata.json
```

### e02-estimate

GitHub-hosted CPUでKenLMを実行します。

Output:

```text
artifacts/lm/ja-6gram.arpa
artifacts/lm/ja-6gram.binary
artifacts/lm/estimation-metadata.json
```

### e02-pack

Vastのexact NeMo runtimeでARPAをNGPU-LM `.nemo`へpackします。

Output:

```text
artifacts/lm/ja-6gram.nemo
artifacts/lm/package-metadata.json
```

### E02 decode

Vastで `common + e02-pack` をrestoreし、`results/E02_ngpulm.jsonl` を生成します。

## 11. E03 — GPU phrase/context biasing

Executor: **Vast GPU**

Inputs: `common`, `e02-pack`

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

追加heavy imageは不要で、canonical NeMo runtimeのboosting-tree pathを使用します。

## 12. E04 — hybrid CTC N-best rerank

Executor: **Vast GPU**

Inputs: `common`, `e02-pack`

Outputs:

```text
results/E04_nbest.jsonl
results/E04_ctc_rerank.jsonl
```

locked modelのCTC branchを使用するためGPU taskです。E05 CPU laneの入力にもなります。

## 13. E05 — frozen encoder phoneme CTC rerank

Canonical E05はGPU/CPUへ分割します。

### e05-extract — Vast

Input: `common`

Output:

```text
artifacts/encoder/*.pt
```

0.6B Parakeet encoder forwardだけをVastで実行します。

### e05-phone — GitHub-hosted CPU

Inputs:

```text
common
phase-e04
e05-extract
```

Outputs:

```text
artifacts/phone_vocab.json
artifacts/phone_head.pt
generated/phone_train.jsonl
results/E04_phone_ready.jsonl
results/E05_phone_rerank.jsonl
```

small projectionの学習・CTC scoring・rerankはCPUで処理します。default imageはsource-matched `phone-e05-<GITHUB_SHA>` です。

## 14. E06 — in-beam fusion

Executor: **Vast GPU**

Inputs:

```text
common
e02-pack
e05-extract
e05-phone
```

E06はNeMo decoder内部APIへ入るため、driverを明示します。

```text
E06_DRIVER=/path/inside/image/to/e06_driver.py
JPA_CF_E06_DRIVER_SHA256=<64hex>
```

GitHub Actions inputでは `e06_driver` と `e06_driver_sha256` を指定します。SHA-256が無い場合はVast allocation前にfailします。Vast container内でも実際のdriver fileを `sha256sum` し、指定digestと一致しなければ実験を開始しません。

Output:

```text
results/E06_inbeam.jsonl
```

## 15. canonical workflowの実行順

初回:

```text
1. research-images
2. research-artifacts-cpu task=common
3. research-phase-vast task=E00
4. research-phase-vast task=E01
5. research-phase-vast task=e02-encode
6. research-artifacts-cpu task=e02-estimate
7. research-phase-vast task=e02-pack
8. research-phase-vast task=E02
9. research-phase-vast task=E03
10. research-phase-vast task=E04
11. research-phase-vast task=e05-extract
12. research-artifacts-cpu task=e05-phone
13. E04/E05のgainを確認
14. research-phase-vast task=E06
```

各taskはoutput snapshotが既に存在すればskipされます。Vast workflowは存在確認を**instance作成前**に行います。

## 16. phase開始前のArtifact validation

local stateへsnapshotをrestoreした後、GPUを使う前に検証します。

```bash
uv run --locked --no-sync python scripts/research/check_phase_artifacts.py E04 \
  --state-root /workspace/state
```

phaseごとのrequiresは `configs/research/e00-e06-artifacts.yaml` をsource of truthとします。

## 17. ローカル開発

通常の編集・lint・CPU testは `.devcontainer` を使います。

```bash
mise run ci
```

Dev ContainerへCUDA/NeMoをもう一式導入しません。GPU validationはsource-matched GHCR phase imageをVast、またはNVIDIA Container Toolkitを持つlocal hostから実行します。

## 18. Storage policy

| Plane | 保存するもの | 保存しないもの |
|---|---|---|
| GHCR / Docker Hub | software environment, thin phase/tool image | audio, model artifact, LM, tensors, results |
| HF Bucket | immutable Artifact snapshots, append-only run evidence | compiler/toolchain image |
| GitHub Actions cache | mise/uv等の小型dependency cache | Docker layer, research dataset |
| GitHub workflow artifact | image manifest, Vast control-plane evidence | persistent research state |
| Vast disk | 実行中scratch | canonical long-term state |

この分離により、GHCR/cache容量を圧迫せず、Buildxはshared parent layerとremote immutable imageを活用し、GPU費用はmodel executionが必要なstageだけに限定できます。
