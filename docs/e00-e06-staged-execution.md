# E00–E06 staged execution protocol

This document is the canonical procedure for running the contextual-ASR ladder before E07a.

## 1. Research order

```text
prepare immutable environment/data/LM
  -> E00 TDT greedy
  -> E01 TDT beam
  -> E02 + NGPU-LM (KenLM-derived)
  -> E03 + GPU-PB / TurboBias-style phrase boosting
  -> E04 + hybrid CTC local reranking
  -> collect category-aware Parquet + Oracle@K
  -> E05 evidence gate
       | fail: stop and analyze candidate generation / semantic errors
       | pass: train frozen-encoder phone head and run E05
  -> E06 only after E05 shows a reproducible held-out gain and a NeMo-3.0.0-specific driver is supplied
```

Do not run E05 merely because the script exists. E05 is justified only when E04 leaves recoverable near-homophone errors: the correct entity survives in the N-best set but is not ranked first.

## 2. Authoritative assets

- base model: `nvidia/parakeet-tdt_ctc-0.6b-ja`, pinned by `locks/hf-revisions.lock.json`
- accepted research model artifact: `saeeew/J-PACF-YOMI-tdt`
- benchmark: `saeeew/JP-HomophoneBench`
- executable benchmark config: `homophone8-audio:test`
- experiment evidence: `hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/runs/<run-id>`
- NeMo Speech source for KenLM tooling: tag `v3.0.0`, commit `fd6a877539710e2b98f28c43272ff81312f83417`
- KenLM source: commit `4cb443e60b7bf2c0ddf3c745378f76cb59e254e5`

The source revisions are recorded in `locks/e00-e06-tools.lock.json`.

## 3. WSL2/self-hosted prerequisites

The repository contract is Linux x86_64 + CUDA 13. The runner labels are:

```text
self-hosted, linux, x64, cuda13
```

Install build dependencies inside WSL2 Ubuntu:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  libboost-all-dev \
  libbz2-dev \
  liblzma-dev \
  zlib1g-dev \
  libsndfile1 \
  ffmpeg \
  jq \
  zstd
```

Do not install a Linux NVIDIA display driver inside WSL2. GPU exposure comes from the Windows NVIDIA driver.

Verify first:

```bash
nvidia-smi
mise install --locked
mise run deps:sync-gpu
uv run --locked python scripts/repro/verify_platform.py --require-gpu
```

The validator requires the repository's current authoritative runtime, including NeMo 3.0.0 and PyTorch 2.12.0+cu132.

## 4. One-time preparation for E00–E04

Run:

```bash
bash scripts/research/prepare_e00_e04.sh
```

The preparation step:

1. materializes the locked GPU environment;
2. resolves the benchmark only at its locked 40-character HF revision;
3. materializes `homophone8-audio:test` into executable local audio;
4. validates exact/near acoustic coverage;
5. verifies and materializes the locked Parakeet `.nemo` artifact;
6. checks out the pinned NeMo Speech source;
7. checks out/builds pinned KenLM;
8. trains the 6-gram BPE KenLM/NGPU-LM when `artifacts/lm/ja-6gram.nemo` is missing.

Expected outputs:

```text
data/generated/eval/
├── bench_index.jsonl
├── nemo_eval.jsonl
├── context_phrases.txt
├── lm_corpus.txt
├── eval_provenance.json
└── audio_coverage.json

artifacts/model/parakeet-tdt_ctc-0.6b-ja.nemo
artifacts/lm/ja-6gram
artifacts/lm/ja-6gram.nemo
.vendor/nemo-speech/
.vendor/kenlm/
```

The `.vendor` directories and HF/model caches should be on the WSL2 Linux filesystem, not `/mnt/c`.

## 5. Run E00–E04 first

Canonical local command:

```bash
export EVAL_DIR=data/generated/eval
export RESULTS_DIR=results/staged
export RUN_E05=0
export RUN_E06=0
bash experiments/run_staged_e00_e06.sh
```

This runs E00–E04 and writes:

```text
results/staged/
├── E00_tdt_greedy.jsonl
├── E01_tdt_beam.jsonl
├── E02_ngpulm.jsonl
├── E03_gpu_pb.jsonl
├── E04_nbest.jsonl
├── E04_ctc_rerank.jsonl
├── metrics_e00_e04.parquet
├── summary_e00_e04.json
└── e05_gate.json
```

The Parquet table contains per-row/category evidence including CER, entity correctness, MRR and Oracle@1/4/8/16/32.

## 6. Interpret E00–E04

Primary comparisons:

### E01 vs E00

Measures ordinary beam-search headroom.

### E02 vs E01

Measures language-model contribution. Pay particular attention to `exact_homophone` and `semantic_only`.

### E03 vs E02

Measures GPU-PB/TurboBias-style contextual boosting. Compare both `near_homophone` and `exact_homophone`; also inspect distractor/bias false positives.

### E04 vs E03

Measures whether the auxiliary CTC branch can improve ranking using independent acoustic-token evidence.

Always compare Top-1 accuracy with Oracle@K. If Oracle@8 is low, the correct candidate is absent and later rerankers cannot recover it.

## 7. E05 gate

`evaluate_e05_gate.py` treats E05 as justified only when `near_homophone` has recoverable ranking errors after E04.

Default rule:

```text
near_homophone count >= 10
Oracle@8 - entity_accuracy >= 0.03
at least one Top1-wrong / Oracle@8-correct row
Wilson 95% lower bound for recoverable-rate > 0
```

The exact-homophone result is reported as a semantic control but does not block E05; identical phone strings are not expected to be fixed by a phone scorer.

Inspect:

```bash
cat results/staged/e05_gate.json | jq
```

If the decision is `stop_before_e05`, do not train the phone head for the primary experiment. You may still use `RUN_E05=force` only as a separately documented ablation.

## 8. Conditional E05

To let the runner obey the gate automatically:

```bash
export RUN_E05=auto
bash experiments/run_staged_e00_e06.sh
```

If the gate passes, the runner performs four additional steps automatically:

1. capture frozen Parakeet/FastConformer encoder states for every runnable benchmark row;
2. join E04 N-best hypotheses with benchmark surface/readings and add candidate `phone_ids`;
3. crop target training features using the E04 CTC window and train only `PhoneCTCHead`;
4. rerank the unchanged E04 candidates with the phone scorer.

Artifacts:

```text
artifacts/encoder/<benchmark-id>.pt
artifacts/encoder_train/<benchmark-id>.pt
artifacts/phone_vocab.json
artifacts/phone_head.pt
data/generated/phone_train.jsonl
results/staged/E04_phone_ready.jsonl
results/staged/E05_phone_rerank.jsonl
results/staged/metrics_e00_e05.parquet
results/staged/summary_e00_e05.json
```

The base Parakeet checkpoint stays frozen.

## 9. E06

E06 is deliberately not an automatic continuation. It changes search itself and reaches into NeMo decoder internals.

Only proceed if E05 demonstrates a reproducible held-out gain and acceptable runtime/damage metrics. Then provide a driver that implements the contract in `patches/README.md` for exactly NeMo Speech v3.0.0 / commit `fd6a877...`:

```bash
export E06_DRIVER=patches/nemo-<sha>/inbeam_driver.py
export RUN_E05=auto
export RUN_E06=1
bash experiments/run_staged_e00_e06.sh
```

E06 must gate expensive CTC/phone scoring by an active context phrase/entity state. Do not run the neural phone scorer for every vocabulary token/frame.

## 10. GitHub Actions on the WSL2 runner

Use:

```text
Actions -> e00-e06-staged-gpu -> Run workflow
```

Recommended first run:

```text
run_e05: auto
run_e06: false
results_name: first-staged
```

The workflow prepares all E00–E04 dependencies, runs the staged ladder, stores Parquet/JSON evidence as a GitHub artifact, and appends a run bundle under:

```text
hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/runs/gh-<run>-<attempt>-e00-e06
```

The Bucket run becomes the immutable source evidence later consumed by E07a when an E05 result exists.

## 11. What to decide after the first run

Use the evidence, not the experiment number:

- E03 improves near/exact but Bias-FPR rises sharply -> boosting saturation; do not just increase PB alpha.
- E04 improves near-homophone -> CTC acoustic evidence is useful.
- E04 has high near Oracle@8 but lower Top-1 -> E05 is a justified phoneme-reranking experiment.
- E04 has low near Oracle@8 -> improve candidate generation/search; a phone reranker cannot recover missing hypotheses.
- exact/semantic-only remain while near improves -> investigate KenLM/semantic entity scoring/PARCO-like contextualization rather than increasing phone weight.
- E05 improves held-out near/voicing/long-vowel/geminate/moraic-nasal with low damage -> only then consider E06 in-beam promotion.

E07a remains separate and should consume an immutable E05 Bucket `source_run_id`; it is not part of the E00–E06 causal ladder.
