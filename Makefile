PYTHON ?= python
RELEASE ?= data/releases/v0.1.0
HF_REPO ?= saeeew/JP-HomophoneBench
HF_CONFIG ?= homophone8-research
HF_LICENSE_POLICY ?= research

.PHONY: lint test validate bench bench-validate hf-publish run-e00 run-e01 run-e02 run-e03 run-e04 run-e05 run-e06

lint:
	ruff check src scripts tests

test:
	pytest

validate:
	$(PYTHON) scripts/validate_manifest.py data/generated/bench.jsonl

bench:
	$(PYTHON) scripts/build_jp_homophone_bench.py \
		--output-dir $(RELEASE) \
		--semantic-tsv data/seed/semantic_homophones.example.tsv

bench-validate:
	$(PYTHON) scripts/validate_jp_homophone_release.py \
		--release-dir $(RELEASE) \
		--schema schemas/benchmark.schema.json \
		--require-core8

hf-publish:
	$(PYTHON) scripts/publish_hf_dataset.py \
		--release-dir $(RELEASE) \
		--repo-id $(HF_REPO) \
		--config-name $(HF_CONFIG) \
		--license-policy $(HF_LICENSE_POLICY)

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
