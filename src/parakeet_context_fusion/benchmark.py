from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .japanese_g2p import reading_to_phones
from .phone_distance import VOICING_PAIRS, weighted_phone_distance

CORE_CATEGORIES = (
    "exact_homophone",
    "near_homophone",
    "voicing",
    "long_vowel",
    "geminate",
    "moraic_nasal",
    "pitch_accent",
    "semantic_only",
)


@dataclass(frozen=True)
class Relation:
    category: str
    phone_distance: float
    reason: str


def _remove(phone: str, sequence: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(item for item in sequence if item != phone)


def _voicing_only(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if len(left) != len(right) or left == right:
        return False
    changed = 0
    for a, b in zip(left, right, strict=True):
        if a == b:
            continue
        if frozenset((a, b)) not in VOICING_PAIRS:
            return False
        changed += 1
    return changed > 0


def classify_relation(
    target_phones: tuple[str, ...],
    candidate_phones: tuple[str, ...],
    *,
    near_threshold: float = 1.0,
) -> Relation:
    distance = weighted_phone_distance(target_phones, candidate_phones)
    if target_phones == candidate_phones:
        return Relation("exact_homophone", 0.0, "identical phone sequence")
    if _remove(":", target_phones) == _remove(":", candidate_phones):
        return Relation("long_vowel", distance, "difference explained by long-vowel marker")
    if _remove("Q", target_phones) == _remove("Q", candidate_phones):
        return Relation("geminate", distance, "difference explained by geminate marker")
    if _remove("N", target_phones) == _remove("N", candidate_phones):
        return Relation("moraic_nasal", distance, "difference explained by moraic nasal")
    if _voicing_only(target_phones, candidate_phones):
        return Relation("voicing", distance, "all substitutions are voicing pairs")
    if distance <= near_threshold:
        return Relation("near_homophone", distance, "weighted phone distance within threshold")
    return Relation("unrelated", distance, "outside near-homophone threshold")


def classify_readings(
    target_reading: str,
    candidate_reading: str,
    *,
    near_threshold: float = 1.0,
) -> Relation:
    return classify_relation(
        reading_to_phones(target_reading),
        reading_to_phones(candidate_reading),
        near_threshold=near_threshold,
    )


def stable_split(
    group_id: str,
    *,
    seed: str = "jp-homophone-bench-v1",
    train: float = 0.70,
    validation: float = 0.10,
) -> str:
    if not 0 <= train <= 1 or not 0 <= validation <= 1 or train + validation > 1:
        raise ValueError("invalid split ratios")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    if fraction < train:
        return "train"
    if fraction < train + validation:
        return "validation"
    return "test"


def make_group_id(*parts: str | None) -> str:
    normalized = "\u241f".join((part or "").strip() for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def difficulty_vector(
    *,
    category: str,
    phone_distance: float | None,
    candidate_count: int,
    has_context: bool,
    snr_db: float | None = None,
) -> dict[str, float | None]:
    if phone_distance is None:
        acoustic = 0.5
    elif phone_distance == 0:
        acoustic = 1.0
    else:
        acoustic = max(0.0, min(1.0, 1.0 - phone_distance / 2.0))
    if snr_db is not None:
        noise_penalty = max(0.0, min(1.0, (20.0 - snr_db) / 20.0))
        acoustic = min(1.0, 0.75 * acoustic + 0.25 * noise_penalty)
    lexical = min(1.0, math.log2(max(2, candidate_count + 1)) / 4.0)
    context_defaults = {
        "semantic_only": 1.0,
        "exact_homophone": 0.9,
        "pitch_accent": 0.45,
        "near_homophone": 0.55,
        "voicing": 0.30,
        "long_vowel": 0.30,
        "geminate": 0.30,
        "moraic_nasal": 0.30,
    }
    context = context_defaults.get(category, 0.5)
    if not has_context and category in {"semantic_only", "exact_homophone"}:
        context = 1.0
    return {
        "acoustic": round(acoustic, 6),
        "lexical": round(lexical, 6),
        "context": round(context, 6),
        "phone_distance": None if phone_distance is None else round(phone_distance, 6),
    }


def canonical_key(record: dict[str, Any]) -> str:
    target = record.get("target") or {}
    candidate_surfaces = sorted(str(item.get("surface", "")) for item in record.get("candidates", []))
    return "|".join([
        str(record.get("category", "")),
        str(target.get("surface", "")),
        str(target.get("reading", "")),
        ",".join(candidate_surfaces),
        str(record.get("text", "")),
    ])
