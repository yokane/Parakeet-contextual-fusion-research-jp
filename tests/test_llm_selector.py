from __future__ import annotations

import pytest

from parakeet_context_fusion.llm_selector import (
    build_messages,
    context_from_row,
    parse_selection,
    prompt_sha256,
    stable_prompt_candidates,
)


def test_stable_prompt_order_is_deterministic() -> None:
    candidates = [{"text": "候補A"}, {"text": "候補B"}, {"text": "候補C"}]
    first = stable_prompt_candidates(candidates, row_id="row-1", top_k=3, seed=7)
    second = stable_prompt_candidates(candidates, row_id="row-1", top_k=3, seed=7)
    assert first == second
    assert sorted(item.original_index for item in first) == [0, 1, 2]


def test_asr_order_can_be_preserved_explicitly() -> None:
    candidates = [{"text": "A"}, {"text": "B"}]
    ordered = stable_prompt_candidates(candidates, row_id="x", top_k=2, seed=7, order="asr")
    assert [item.original_index for item in ordered] == [0, 1]


def test_parse_selection_accepts_json_and_rejects_out_of_range() -> None:
    assert parse_selection('{"selected": 1}', candidate_count=3) == 1
    assert parse_selection('```json\n{"selected": 2}\n```', candidate_count=3) == 2
    with pytest.raises(ValueError):
        parse_selection('{"selected": 3}', candidate_count=3)


def test_reference_context_is_forbidden() -> None:
    row = {
        "text": "正解文",
        "metadata": {
            "document_context": "許可文脈",
            "gold_text": "別名の正解文",
        },
    }
    with pytest.raises(ValueError):
        context_from_row(row, "text")
    with pytest.raises(ValueError):
        context_from_row(row, "metadata.gold_text")
    assert context_from_row(row, "metadata.document_context") == "許可文脈"


def test_prompt_hash_changes_with_candidates() -> None:
    a = stable_prompt_candidates([{"text": "気候"}, {"text": "機構"}], row_id="a", top_k=2, seed=7)
    b = stable_prompt_candidates([{"text": "気候"}, {"text": "紀行"}], row_id="a", top_k=2, seed=7)
    a_messages = build_messages(a)
    b_messages = build_messages(b)
    assert prompt_sha256(a_messages) != prompt_sha256(b_messages)
