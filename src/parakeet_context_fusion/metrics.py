from __future__ import annotations

from collections.abc import Sequence


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(list(reference), list(hypothesis)) / len(reference)


def entity_exact_match(target: str, hypothesis: str) -> bool:
    return target in hypothesis


def reciprocal_rank(target: str, ranked_texts: Sequence[str]) -> float:
    for rank, text in enumerate(ranked_texts, 1):
        if target in text:
            return 1.0 / rank
    return 0.0


def oracle_at_k(target: str, ranked_texts: Sequence[str], k: int) -> bool:
    return any(target in text for text in ranked_texts[:k])


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def bias_false_positive_rate(false_insertions: int, negative_utterances: int) -> float:
    return false_insertions / negative_utterances if negative_utterances else 0.0
