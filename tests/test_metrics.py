from parakeet_context_fusion.metrics import cer, oracle_at_k, reciprocal_rank
from parakeet_context_fusion.phoneme import weighted_phone_distance


def test_cer_identity() -> None:
    assert cer("気候", "気候") == 0.0


def test_cer_substitution() -> None:
    assert cer("気候", "機構") == 1.0


def test_rank_metrics() -> None:
    ranked = ["新しい機構モデル", "新しい気候モデル"]
    assert reciprocal_rank("気候", ranked) == 0.5
    assert not oracle_at_k("気候", ranked, 1)
    assert oracle_at_k("気候", ranked, 2)


def test_exact_homophone_phone_distance_is_zero() -> None:
    phones = ["k", "i", "k", "o", ":"]
    assert weighted_phone_distance(phones, phones) == 0.0


def test_voicing_is_cheaper_than_generic_substitution() -> None:
    voiced = weighted_phone_distance(["k", "a"], ["g", "a"])
    generic = weighted_phone_distance(["k", "a"], ["m", "a"])
    assert voiced < generic
