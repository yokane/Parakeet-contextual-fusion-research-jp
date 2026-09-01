PYTHON ?= python
RELEASE ?= data/releases/v0.1.0
HF_REPO ?= saeeew/JP-HomophoneBench
HF_MODEL_REPO ?= saeeew/J-PACF-YOMI-tdt
HF_BUCKET ?= saeeew/J-PACF-YOMI-tdt-bucket
HF_CONFIG ?= homophone8
HF_SPLIT ?= test
HF_LICENSE_POLICY ?= permissive
GHCR_IMAGE ?= ghcr.io/yokane/jpacf-yomi-tdt-runtime
GENERATED ?= data/generated
RESULTS_DIR ?= results
RESULT_SPECS ?=
SATURATION_SPEC ?= configs/saturation.example.json
COVERAGE_PROVENANCE ?= $(GENERATED)/eval_provenance.json
REQUIRED_AUDIO_CATEGORIES ?= exact_homophone near_homophone
MIN_AUDIO_PER_CATEGORY ?= 5
DISTRACTOR_COUNTS ?= 0,10,100
CONTEXT_STRESS_DIR ?= $(GENERATED)/context-stress
CONTEXT_STRESS_RESULTS ?= $(RESULTS_DIR)/context-stress
CANDIDATE_DIR ?=
RUN_DIR ?=

.PHONY: lint test validate bench bench-permissive bench-validate hf-publish hf-eval-index hf-eval-audio metrics saturation audio-coverage context-stress-lists context-stress-analyze hf-bucket-bootstrap hf-bucket-validate hf-candidate-push hf-run-push docker-build docker-pull run-e00 run-e01 run-e02 run-e03 run-e04 run-e05 run-e06

lint:
	ruff check src scripts tests

test:
	pytest

validate:
	$(PYTHON) scripts/validate_eval_manifest.py $(GENERATED)/bench_index.jsonl

bench:
	$(PYTHON) scripts/build_jp_homophone_bench.py \
		--output-dir $(RELEASE) \
		--semantic-tsv data/seed/semantic_homophones.example.tsv

bench-permissive: bench
	$(PYTHON) scripts/augment_permissive_core8.py \
		--release-dir $(RELEASE) \
		--seed data/seed/permissive_phonetic_core.tsv

bench-validate:
	$(PYTHON) scripts/validate_jp_homophone_release.py \
		--release-dir $(RELEASE) \
		--schema schemas/benchmark.schema.json \
		--require-core8

hf-publish:
	@if [ "$(HF_LICENSE_POLICY)" = "permissive" ]; then \
		$(MAKE) bench-permissive RELEASE=$(RELEASE); \
	else \
		$(MAKE) bench RELEASE=$(RELEASE); \
	fi
	$(MAKE) bench-validate RELEASE=$(RELEASE)
	$(PYTHON) scripts/publish_hf_dataset.py \
		--release-dir $(RELEASE) \
		--repo-id $(HF_REPO) \
		--config-name $(HF_CONFIG) \
		--license-policy $(HF_LICENSE_POLICY)

hf-eval-index:
	$(PYTHON) scripts/materialize_hf_eval.py \
		--repo-id $(HF_REPO) \
		--config $(HF_CONFIG) \
		--split $(HF_SPLIT) \
		--output-dir $(GENERATED) \
		--no-rehydrate-audio
	$(PYTHON) scripts/validate_eval_manifest.py $(GENERATED)/bench_index.jsonl

hf-eval-audio:
	$(PYTHON) scripts/materialize_hf_eval.py \
		--repo-id $(HF_REPO) \
		--config $(HF_CONFIG) \
		--split $(HF_SPLIT) \
		--output-dir $(GENERATED) \
		--rehydrate-audio \
		--require-audio
	$(PYTHON) scripts/validate_eval_manifest.py $(GENERATED)/nemo_eval.jsonl --require-audio

metrics:
	@test -n "$(RESULT_SPECS)" || { echo 'Set RESULT_SPECS="E00=path E01=path ..."' >&2; exit 2; }
	$(PYTHON) scripts/collect_experiment_metrics.py \
		--benchmark $(GENERATED)/bench_index.jsonl \
		$(foreach spec,$(RESULT_SPECS),--result $(spec)) \
		--parquet $(RESULTS_DIR)/metrics.parquet \
		--summary $(RESULTS_DIR)/summary.json

saturation:
	$(PYTHON) scripts/analyze_saturation.py \
		--metrics $(RESULTS_DIR)/metrics.parquet \
		--sweep $(SATURATION_SPEC) \
		--output $(RESULTS_DIR)/saturation.json

audio-coverage:
	$(PYTHON) scripts/validate_audio_coverage.py \
		--provenance $(COVERAGE_PROVENANCE) \
		--min-per-category $(MIN_AUDIO_PER_CATEGORY) \
		--min-total $(MIN_AUDIO_PER_CATEGORY) \
		$(foreach category,$(REQUIRED_AUDIO_CATEGORIES),--required-category $(category)) \
		--output $(RESULTS_DIR)/audio_coverage.json

context-stress-lists:
	$(PYTHON) scripts/build_context_stress.py \
		--benchmark $(CONTEXT_STRESS_DIR)/bench_index.jsonl \
		--execution-manifest $(CONTEXT_STRESS_DIR)/nemo_eval.jsonl \
		--output-dir $(CONTEXT_STRESS_DIR)/lists \
		--distractor-counts $(DISTRACTOR_COUNTS) \
		--seed 20260901

context-stress-analyze:
	$(PYTHON) scripts/analyze_context_stress.py \
		--metrics $(CONTEXT_STRESS_RESULTS)/metrics.parquet \
		--stress-manifest $(CONTEXT_STRESS_DIR)/lists/context_stress_manifest.json \
		--output $(CONTEXT_STRESS_RESULTS)/context_stress.json

hf-bucket-bootstrap:
	HF_BUCKET=$(HF_BUCKET) bash scripts/hf/hf-bootstrap-bucket.sh

hf-bucket-validate:
	@tmp="$$(mktemp)"; trap 'rm -f "$$tmp"' EXIT; \
	hf buckets list "$(HF_BUCKET)" -R -q --token "$$HF_TOKEN" > "$$tmp"; \
	$(PYTHON) scripts/hf/validate_bucket_layout.py --listing "$$tmp"

hf-candidate-push:
	@test -n "$(CANDIDATE_DIR)" || { echo 'Set CANDIDATE_DIR=dist/hf-candidate/<release>' >&2; exit 2; }
	HF_BUCKET=$(HF_BUCKET) bash scripts/hf/hf-push-candidate.sh $(CANDIDATE_DIR)

hf-run-push:
	@test -n "$(RUN_DIR)" || { echo 'Set RUN_DIR=dist/hf-runs/<run-id>' >&2; exit 2; }
	HF_BUCKET=$(HF_BUCKET) bash scripts/hf/hf-push-run.sh $(RUN_DIR)

docker-build:
	DOCKER_BUILDKIT=1 docker build --progress=plain -t $(GHCR_IMAGE):local .

docker-pull:
	docker pull $(GHCR_IMAGE):main

run-e00:
	bash experiments/E00_tdt_greedy.sh

run-e01:
	bash experiments/E01_tdt_beam.sh

run-e02:
	bash experiments/E02_ngpulm.sh

run-e03:
	bash experiments/E03_gpu_pb.sh

run-e04:
	bash experiments/E04_ctc_rerank.sh

run-e05:
	bash experiments/E05_phone_rerank.sh

run-e06:
	bash experiments/E06_inbeam.sh
