from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_CRITICAL_FILES = (
    "scripts/research/prepare_e00_e04.sh",
    "scripts/train_ngpulm.sh",
    "experiments/_common.sh",
    "experiments/E04_ctc_rerank.sh",
    "experiments/E05_phone_rerank.sh",
    "experiments/E06_inbeam.sh",
    "experiments/run_staged_e00_e06.sh",
    ".github/workflows/e00-e06-staged-gpu.yml",
)


def test_gpu_critical_uv_runs_do_not_resync_environment() -> None:
    unsafe = "uv run --locked python"
    for relative in GPU_CRITICAL_FILES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert unsafe not in text, f"{relative} may drop the synced GPU extras; use --no-sync"


def test_gpu_sync_task_is_not_source_cached() -> None:
    text = (ROOT / "mise.toml").read_text(encoding="utf-8")
    section = text.split('[tasks."deps:sync-gpu"]', 1)[1].split("[tasks.", 1)[0]
    assert "uv sync --locked --extra dev --extra gpu" in section
    assert "sources =" not in section
