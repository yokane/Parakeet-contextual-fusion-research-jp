from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


SYSTEM_PROMPT = """あなたは日本語ASRのN-best候補選択器です。
与えられた候補の中から、発話として最も妥当なものを必ず1つだけ選んでください。
候補を書き換えたり、新しい文を生成したりしてはいけません。
正解文・参照文は与えられていません。候補内部の日本語文脈だけを根拠に判断してください。
出力は JSON オブジェクト {\"selected\": <候補ID>} のみとしてください。説明は不要です。"""

_FORBIDDEN_CONTEXT_FIELDS = {
    "text",
    "reference",
    "reference_text",
    "gold",
    "gold_text",
    "target_text",
}


@dataclass(frozen=True)
class PromptCandidate:
    prompt_id: int
    original_index: int
    text: str


def _stable_key(*, row_id: str, seed: int, original_index: int, text: str) -> str:
    value = f"{seed}\0{row_id}\0{original_index}\0{text}".encode()
    return hashlib.sha256(value).hexdigest()


def stable_prompt_candidates(
    candidates: list[dict[str, Any]],
    *,
    row_id: str,
    top_k: int,
    seed: int,
    order: str = "stable_shuffle",
) -> list[PromptCandidate]:
    limited = list(enumerate(candidates[:top_k]))
    if order == "stable_shuffle":
        limited.sort(
            key=lambda pair: _stable_key(
                row_id=row_id,
                seed=seed,
                original_index=pair[0],
                text=str(pair[1].get("text") or ""),
            )
        )
    elif order != "asr":
        raise ValueError(f"unsupported candidate order: {order}")
    return [
        PromptCandidate(prompt_id=prompt_id, original_index=index, text=str(candidate.get("text") or ""))
        for prompt_id, (index, candidate) in enumerate(limited)
    ]


def context_from_row(row: dict[str, Any], field: str | None) -> str | None:
    if not field:
        return None
    parts = field.split(".")
    if any(part.lower() in _FORBIDDEN_CONTEXT_FIELDS for part in parts):
        raise ValueError(f"context field {field!r} is forbidden because it can leak the benchmark reference")
    value: Any = row
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"context field {field!r} must resolve to a string")
    return value.strip() or None


def build_messages(
    prompt_candidates: list[PromptCandidate],
    *,
    external_context: str | None = None,
) -> list[dict[str, str]]:
    lines = ["候補一覧:"]
    for candidate in prompt_candidates:
        lines.append(f"[{candidate.prompt_id}] {candidate.text}")
    if external_context:
        lines.extend(["", "許可された外部文脈:", external_context])
    lines.extend(["", 'JSONのみを返してください。例: {"selected": 2}'])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def prompt_sha256(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_selection(raw_output: str, *, candidate_count: int) -> int:
    text = raw_output.strip()
    payload: Any
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("selector output did not contain a JSON object") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("selector output contained invalid JSON") from exc
    if not isinstance(payload, dict) or type(payload.get("selected")) is not int:
        raise ValueError("selector output must be an object containing integer field 'selected'")
    selected = int(payload["selected"])
    if selected < 0 or selected >= candidate_count:
        raise ValueError(f"selected candidate {selected} is outside 0..{candidate_count - 1}")
    return selected
