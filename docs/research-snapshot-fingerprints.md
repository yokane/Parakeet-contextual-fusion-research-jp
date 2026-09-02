# E00–E06 content-addressed snapshot fingerprints

E00–E06 の再利用可能Artifactは、benchmark/model revisionだけではなく、各stageの実装identityも含めて判定します。これにより実装を変更したのに古いHF Bucket snapshotを誤ってcache hitすることを防ぎつつ、無関係なstageまで再計算することを避けます。

## Identityの2層

`research_key` はデータ/model側の固定identityです。

```text
v1-bench-<benchmark-revision>-model-<model-revision>-ng6
```

各snapshotはさらに `stage-fingerprint` を持ちます。

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/
  workspace-cache/e00-e06/
    <research-key>/
      <stage>/
        <stage-fingerprint>/
```

`stage-fingerprint` はSHA-256で、次をcanonical JSONへ正規化して計算します。

- `configs/research/e00-e06-artifacts.yaml` のtask/output/publish契約
- taskごとの `fingerprint_sources`
- taskごとの固定parameter (`fingerprint_values`)
- upstream stageのfingerprint
- 必要なexternal identity (`fingerprint_external`)

計算実装は `scripts/research/stage_fingerprints.py` です。

```bash
uv run --locked --no-sync python scripts/research/stage_fingerprints.py
uv run --locked --no-sync python scripts/research/stage_fingerprints.py --task e05-phone --field fingerprint
```

## Selective + transitive invalidation

例えば `scripts/rerank_phone.py` だけを変更した場合、次のようになります。

```text
common          reuse
e02-*           reuse
E00-E04         reuse
e05-extract     reuse
e05-phone       invalidate
E06             invalidate  <- upstream e05-phone fingerprintを含むため
```

一方、`scripts/research/prepare_common_artifacts.sh` を変更した場合は `common` fingerprintが変わるため、`common`を入力に持つ全下流stageが連鎖的にinvalidateされます。

repository全体のGit SHAを単純にsnapshot keyへ入れないのは、READMEやDev Containerだけの変更でKenLMやGPU decodeを再実行しないためです。

## Hosted workflowからVastへのidentity固定

GitHub-hosted control planeで全stage fingerprintを1回計算し、base64 JSONとして

```text
JPA_CF_STAGE_FINGERPRINTS_B64
```

へ格納します。同じ値をVast containerへ渡すことで、host側の「snapshot exists判定」とGPU container側の「pull/push先」が必ず一致します。

これは特にPR branchで重要です。GPU taskはdefaultでmutable `phase-*-current` を使用せず、

```text
phase-<stage>-${GITHUB_SHA}
```

というsource-matched immutable tagを解決します。該当imageがGHCRになければ、設定済みの場合のみ同一tagをpublic Docker Hubから解決します。どちらにも存在しない場合はGPUを借りる前にfailします。

## E02

E02のfingerprintは処理境界ごとに独立します。

```text
common
  -> e02-encode
       -> e02-estimate
            -> e02-pack
                 -> E02/E03/E04
```

KenLM CPU estimationは以下を明示identityに含めます。

```text
KenLM revision: 4cb443e60b7bf2c0ddf3c745378f76cb59e254e5
N-gram order:   6
```

したがってE05だけを修正してもKenLMは再生成されません。

## E05

E05は

```text
common + phase-e04 + e05-extract -> e05-phone
```

のfingerprint lineageを持ちます。phone-head/reranker実装変更は`e05-phone`とそのconsumer E06のみをinvalidateします。

CPU imageもmutable current aliasではなく、defaultで

```text
ghcr.io/yokane/jpacf-yomi-tdt-tools:phone-e05-${GITHUB_SHA}
```

を使用します。

## E06 external driver

E06はNeMo内部APIへ依存するdriverを外部指定できるため、pathだけではsnapshot identityとして不十分です。canonical E06 workflowでは次の2値を要求します。

```text
E06_DRIVER=/path/inside/image/to/driver.py
JPA_CF_E06_DRIVER_SHA256=<64-hex SHA-256>
```

GitHub Actions inputでは `e06_driver` と `e06_driver_sha256` に対応します。SHA-256がない、または64桁hexでない場合はVast allocation前にfailします。

custom E06 imageを使う場合は、driverをimage内に固定してから次を計算してください。

```bash
sha256sum path/to/e06_driver.py
```

driverの内容が変わるとE06 fingerprintも変わるため、古いE06 snapshotは再利用されません。

## Snapshot manifest

各delta snapshotにはfingerprintを含むmanifestを保存します。

```text
.jpacf-snapshots/<stage>-<fingerprint-prefix>.json
```

主なfield:

```json
{
  "research_key": "...",
  "stage": "phase-e04",
  "fingerprint": "<64hex>",
  "input_refs": ["common/<64hex>", "e02-pack/<64hex>"],
  "output_ref": "phase-e04/<64hex>",
  "source_git_sha": "...",
  "files": []
}
```

Artifact file自体のSHA-256 inventoryも従来通り保持します。

## 研究時の確認

GPUを確保する前にplanを確認できます。

```bash
export HF_TOKEN=hf_...
RESEARCH_KEY="$(uv run --locked --no-sync python scripts/research/research_key.py)"

bash scripts/hf/hf-research-snapshot.sh plan E04
bash scripts/hf/hf-research-snapshot.sh remote "$RESEARCH_KEY" E04
bash scripts/hf/hf-research-snapshot.sh exists "$RESEARCH_KEY" E04
```

`remote`に `<stage>/<stage-fingerprint>` が含まれていることを確認します。

この方式により、HF Bucketは研究Artifactのcontent-addressed lineage、GHCR/Docker Hubはsoftware environment、GitHub Actions cacheはmise/uvなどの小型cache、VastはGPU計算だけ、という責務分離を維持できます。
