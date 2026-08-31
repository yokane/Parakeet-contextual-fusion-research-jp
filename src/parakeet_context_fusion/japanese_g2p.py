from __future__ import annotations

import unicodedata
from functools import lru_cache

_KANA_TO_PHONES: dict[str, tuple[str, ...]] = {
    "ア": ("a",), "イ": ("i",), "ウ": ("u",), "エ": ("e",), "オ": ("o",),
    "カ": ("k", "a"), "キ": ("k", "i"), "ク": ("k", "u"), "ケ": ("k", "e"), "コ": ("k", "o"),
    "ガ": ("g", "a"), "ギ": ("g", "i"), "グ": ("g", "u"), "ゲ": ("g", "e"), "ゴ": ("g", "o"),
    "サ": ("s", "a"), "シ": ("sh", "i"), "ス": ("s", "u"), "セ": ("s", "e"), "ソ": ("s", "o"),
    "ザ": ("z", "a"), "ジ": ("j", "i"), "ズ": ("z", "u"), "ゼ": ("z", "e"), "ゾ": ("z", "o"),
    "タ": ("t", "a"), "チ": ("ch", "i"), "ツ": ("ts", "u"), "テ": ("t", "e"), "ト": ("t", "o"),
    "ダ": ("d", "a"), "ヂ": ("j", "i"), "ヅ": ("z", "u"), "デ": ("d", "e"), "ド": ("d", "o"),
    "ナ": ("n", "a"), "ニ": ("n", "i"), "ヌ": ("n", "u"), "ネ": ("n", "e"), "ノ": ("n", "o"),
    "ハ": ("h", "a"), "ヒ": ("h", "i"), "フ": ("f", "u"), "ヘ": ("h", "e"), "ホ": ("h", "o"),
    "バ": ("b", "a"), "ビ": ("b", "i"), "ブ": ("b", "u"), "ベ": ("b", "e"), "ボ": ("b", "o"),
    "パ": ("p", "a"), "ピ": ("p", "i"), "プ": ("p", "u"), "ペ": ("p", "e"), "ポ": ("p", "o"),
    "マ": ("m", "a"), "ミ": ("m", "i"), "ム": ("m", "u"), "メ": ("m", "e"), "モ": ("m", "o"),
    "ヤ": ("y", "a"), "ユ": ("y", "u"), "ヨ": ("y", "o"),
    "ラ": ("r", "a"), "リ": ("r", "i"), "ル": ("r", "u"), "レ": ("r", "e"), "ロ": ("r", "o"),
    "ワ": ("w", "a"), "ヰ": ("i",), "ヱ": ("e",), "ヲ": ("o",),
    "ヴ": ("v", "u"),
    "キャ": ("ky", "a"), "キュ": ("ky", "u"), "キョ": ("ky", "o"),
    "ギャ": ("gy", "a"), "ギュ": ("gy", "u"), "ギョ": ("gy", "o"),
    "シャ": ("sh", "a"), "シュ": ("sh", "u"), "ショ": ("sh", "o"),
    "ジャ": ("j", "a"), "ジュ": ("j", "u"), "ジョ": ("j", "o"),
    "チャ": ("ch", "a"), "チュ": ("ch", "u"), "チョ": ("ch", "o"),
    "ニャ": ("ny", "a"), "ニュ": ("ny", "u"), "ニョ": ("ny", "o"),
    "ヒャ": ("hy", "a"), "ヒュ": ("hy", "u"), "ヒョ": ("hy", "o"),
    "ビャ": ("by", "a"), "ビュ": ("by", "u"), "ビョ": ("by", "o"),
    "ピャ": ("py", "a"), "ピュ": ("py", "u"), "ピョ": ("py", "o"),
    "ミャ": ("my", "a"), "ミュ": ("my", "u"), "ミョ": ("my", "o"),
    "リャ": ("ry", "a"), "リュ": ("ry", "u"), "リョ": ("ry", "o"),
    "ファ": ("f", "a"), "フィ": ("f", "i"), "フェ": ("f", "e"), "フォ": ("f", "o"),
    "ティ": ("t", "i"), "トゥ": ("t", "u"), "ディ": ("d", "i"), "ドゥ": ("d", "u"),
    "チェ": ("ch", "e"), "シェ": ("sh", "e"), "ジェ": ("j", "e"),
    "ウィ": ("w", "i"), "ウェ": ("w", "e"), "ウォ": ("w", "o"),
    "クァ": ("k", "w", "a"), "クィ": ("k", "w", "i"), "クェ": ("k", "w", "e"), "クォ": ("k", "w", "o"),
    "グァ": ("g", "w", "a"), "グィ": ("g", "w", "i"), "グェ": ("g", "w", "e"), "グォ": ("g", "w", "o"),
    "ヴァ": ("v", "a"), "ヴィ": ("v", "i"), "ヴェ": ("v", "e"), "ヴォ": ("v", "o"),
}

_SPECIAL = {"ッ": ("Q",), "ン": ("N",), "ー": (":",)}
_SMALL_VOWELS = {"ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o"}


def _hira_to_kata(text: str) -> str:
    chars: list[str] = []
    for char in text:
        code = ord(char)
        if 0x3041 <= code <= 0x3096:
            chars.append(chr(code + 0x60))
        else:
            chars.append(char)
    return "".join(chars)


def normalize_reading(reading: str) -> str:
    value = unicodedata.normalize("NFKC", reading).strip().replace(" ", "")
    return _hira_to_kata(value)


@lru_cache(maxsize=32768)
def reading_to_phones(reading: str) -> tuple[str, ...]:
    text = normalize_reading(reading)
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in _SPECIAL:
            result.extend(_SPECIAL[char])
            index += 1
            continue
        matched = False
        for width in (3, 2, 1):
            chunk = text[index : index + width]
            phones = _KANA_TO_PHONES.get(chunk)
            if phones is not None:
                result.extend(phones)
                index += width
                matched = True
                break
        if matched:
            continue
        if char in _SMALL_VOWELS:
            result.append(_SMALL_VOWELS[char])
            index += 1
            continue
        if unicodedata.category(char).startswith(("P", "Z")):
            index += 1
            continue
        result.append(f"UNK:{char}")
        index += 1
    return tuple(result)


def surface_to_reading(surface: str) -> str | None:
    try:
        import pyopenjtalk  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        reading = pyopenjtalk.g2p(surface, kana=True)
    except Exception:
        return None
    if not reading:
        return None
    return normalize_reading(str(reading))


def ensure_reading(surface: str, reading: str | None) -> str | None:
    if reading:
        return normalize_reading(reading)
    return surface_to_reading(surface)
