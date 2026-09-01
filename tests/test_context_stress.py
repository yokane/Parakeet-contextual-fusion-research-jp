from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_context_stress_preserves_required_and_nests_distractors(tmp_path: Path) -> None:
    benchmark = tmp_path / "bench.jsonl"
    execution = tmp_path / "nemo.jsonl"
    output = tmp_path / "stress"
    write_jsonl(
        benchmark,
        [
            {
                "id": "run-1",
                "target": {"surface": "気候"},
                "candidates": [{"surface": "機構"}],
            },
            {
                "id": "lex-1",
                "target": {"surface": "公園"},
                "candidates": [{"surface": "講演"}],
            },
            {
                "id": "lex-2",
                "target": {"surface": "後援"},
                "candidates": [{"surface": "公演"}],
            },
        ],
    )
    write_jsonl(execution, [{"benchmark_id": "run-1"}])

    subprocess.run(
        [
            sys.executable,
            "scripts/build_context_stress.py",
            "--benchmark",
            str(benchmark),
            "--execution-manifest",
            str(execution),
            "--output-dir",
            str(output),
            "--distractor-counts",
            "0,2,4",
        ],
        check=True,
    )

    manifest = json.loads((output / "context_stress_manifest.json").read_text(encoding="utf-8"))
    assert manifest["required_phrase_count"] == 2
    assert [case["actual_distractors"] for case in manifest["cases"]] == [0, 2, 4]

    d0 = set((output / "context_d00000.txt").read_text(encoding="utf-8").splitlines())
    d2 = set((output / "context_d00002.txt").read_text(encoding="utf-8").splitlines())
    d4 = set((output / "context_d00004.txt").read_text(encoding="utf-8").splitlines())
    assert d0 == {"気候", "機構"}
    assert d0 < d2 < d4


def test_context_stress_fails_when_pool_is_too_short(tmp_path: Path) -> None:
    benchmark = tmp_path / "bench.jsonl"
    execution = tmp_path / "nemo.jsonl"
    write_jsonl(
        benchmark,
        [
            {"id": "run", "target": {"surface": "気候"}, "candidates": []},
            {"id": "other", "target": {"surface": "公園"}, "candidates": []},
        ],
    )
    write_jsonl(execution, [{"benchmark_id": "run"}])
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_context_stress.py",
            "--benchmark",
            str(benchmark),
            "--execution-manifest",
            str(execution),
            "--output-dir",
            str(tmp_path / "out"),
            "--distractor-counts",
            "0,10",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "only 1 are available" in result.stderr
