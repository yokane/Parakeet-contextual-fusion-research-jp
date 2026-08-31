from parakeet_context_fusion.benchmark import CORE_CATEGORIES, classify_readings, stable_split
from parakeet_context_fusion.japanese_g2p import normalize_reading, reading_to_phones


def test_reading_normalizes_hiragana_and_katakana() -> None:
    assert normalize_reading("きこう") == "キコウ"
    assert reading_to_phones("きこう") == reading_to_phones("キコウ")
    assert reading_to_phones("きこう") == ("k", "i", "k", "o", "u")


def test_special_moras_are_explicit() -> None:
    assert "Q" in reading_to_phones("カッコ")
    assert "N" in reading_to_phones("カンコ")
    assert ":" in reading_to_phones("カー")


def test_exact_homophone() -> None:
    relation = classify_readings("きこう", "きこう")
    assert relation.category == "exact_homophone"
    assert relation.phone_distance == 0.0


def test_long_vowel_class_precedes_generic_near_homophone() -> None:
    assert classify_readings("カー", "カ").category == "long_vowel"


def test_geminate_class() -> None:
    assert classify_readings("カッコ", "カコ").category == "geminate"


def test_moraic_nasal_class() -> None:
    assert classify_readings("カンコ", "カコ").category == "moraic_nasal"


def test_voicing_class() -> None:
    assert classify_readings("カク", "ガク").category == "voicing"


def test_near_homophone_class() -> None:
    assert classify_readings("カビ", "カミ", near_threshold=1.0).category == "near_homophone"


def test_group_split_is_deterministic() -> None:
    assert stable_split("same-reading-group") == stable_split("same-reading-group")


def test_core8_is_exactly_the_intended_taxonomy() -> None:
    assert set(CORE_CATEGORIES) == {
        "exact_homophone", "near_homophone", "voicing", "long_vowel",
        "geminate", "moraic_nasal", "pitch_accent", "semantic_only",
    }
