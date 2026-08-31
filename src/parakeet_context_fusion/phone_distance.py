from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

VOICING_PAIRS = {
    frozenset(("k", "g")),
    frozenset(("s", "z")),
    frozenset(("sh", "j")),
    frozenset(("t", "d")),
    frozenset(("ch", "j")),
    frozenset(("h", "b")),
    frozenset(("f", "b")),
}
SPECIAL_MORAS = {"N", "Q", ":"}


@dataclass(frozen=True)
class PhoneCosts:
    default: float = 1.0
    voicing: float = 0.25
    special_mora: float = 0.25
    vowel: float = 0.5


DEFAULT_PHONE_COSTS = PhoneCosts()


def substitution_cost(a: str, b: str, costs: PhoneCosts = DEFAULT_PHONE_COSTS) -> float:
    if a == b:
        return 0.0
    if frozenset((a, b)) in VOICING_PAIRS:
        return costs.voicing
    vowels = {"a", "i", "u", "e", "o"}
    if a in vowels and b in vowels:
        return costs.vowel
    if a in SPECIAL_MORAS or b in SPECIAL_MORAS:
        return costs.special_mora
    return costs.default


def insertion_deletion_cost(phone: str, costs: PhoneCosts = DEFAULT_PHONE_COSTS) -> float:
    return costs.special_mora if phone in SPECIAL_MORAS else costs.default


def weighted_phone_distance(
    left: Sequence[str],
    right: Sequence[str],
    costs: PhoneCosts = DEFAULT_PHONE_COSTS,
) -> float:
    previous = [0.0]
    for phone in right:
        previous.append(previous[-1] + insertion_deletion_cost(phone, costs))

    for left_phone in left:
        current = [previous[0] + insertion_deletion_cost(left_phone, costs)]
        for j, right_phone in enumerate(right, 1):
            delete = previous[j] + insertion_deletion_cost(left_phone, costs)
            insert = current[j - 1] + insertion_deletion_cost(right_phone, costs)
            substitute = previous[j - 1] + substitution_cost(left_phone, right_phone, costs)
            current.append(min(delete, insert, substitute))
        previous = current
    return previous[-1]


def relation_for_phones(target: Sequence[str], candidate: Sequence[str]) -> str:
    return "exact_homophone" if list(target) == list(candidate) else "near_homophone"
