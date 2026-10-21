"""Japanese G2P — supports kanji/hiragana/katakana input.

Priority: pyopenjtalk > pykakasi + built-in table > kana-only built-in table
"""

from __future__ import annotations

from dsmaker.g2p.base import G2PBase

# DiffSinger Japanese phoneme set (romaji-based, matching common dictionaries)
JA_PHONEMES = {
    "a", "i", "u", "e", "o",
    "k", "s", "t", "n", "h", "m", "y", "r", "w",
    "g", "z", "d", "b", "p",
    "ky", "sh", "ch", "ny", "hy", "my", "ry",
    "gy", "j", "dy", "by", "py",
    "ts", "f", "v",
    "N",       # nasal coda (ん)
    "cl",      # geminate (っ)
    "SP",      # silence/pause
    "AP",      # aspiration/breath
}

# OpenJTalk phoneme → DiffSinger phoneme
_JTALK_TO_DS = {
    "pau": "SP", "sil": "SP", "cl": "cl",
    "a": "a", "i": "i", "u": "u", "e": "e", "o": "o",
    "A": "a", "I": "i", "U": "u", "E": "e", "O": "o",
    "N": "N",
    "k": "k", "s": "s", "t": "t", "n": "n", "h": "h",
    "m": "m", "y": "y", "r": "r", "w": "w",
    "g": "g", "z": "z", "d": "d", "b": "b", "p": "p",
    "ky": "ky", "sh": "sh", "ch": "ch", "ny": "ny",
    "hy": "hy", "my": "my", "ry": "ry",
    "gy": "gy", "j": "j", "dy": "dy", "by": "by", "py": "py",
    "ts": "ts", "f": "f", "v": "v",
}

# Built-in kana → phoneme table
# Longer patterns must come first for greedy matching.
_KANA_TABLE: list[tuple[str, list[str]]] = [
    # Special
    ("ん", ["N"]), ("っ", ["cl"]),
    # Palatalized (きゃ etc.)
    ("きゃ", ["ky", "a"]), ("きゅ", ["ky", "u"]), ("きょ", ["ky", "o"]),
    ("しゃ", ["sh", "a"]), ("しゅ", ["sh", "u"]), ("しょ", ["sh", "o"]),
    ("ちゃ", ["ch", "a"]), ("ちゅ", ["ch", "u"]), ("ちょ", ["ch", "o"]),
    ("にゃ", ["ny", "a"]), ("にゅ", ["ny", "u"]), ("にょ", ["ny", "o"]),
    ("ひゃ", ["hy", "a"]), ("ひゅ", ["hy", "u"]), ("ひょ", ["hy", "o"]),
    ("みゃ", ["my", "a"]), ("みゅ", ["my", "u"]), ("みょ", ["my", "o"]),
    ("りゃ", ["ry", "a"]), ("りゅ", ["ry", "u"]), ("りょ", ["ry", "o"]),
    ("ぎゃ", ["gy", "a"]), ("ぎゅ", ["gy", "u"]), ("ぎょ", ["gy", "o"]),
    ("じゃ", ["j", "a"]),  ("じゅ", ["j", "u"]),  ("じょ", ["j", "o"]),
    ("びゃ", ["by", "a"]), ("びゅ", ["by", "u"]), ("びょ", ["by", "o"]),
    ("ぴゃ", ["py", "a"]), ("ぴゅ", ["py", "u"]), ("ぴょ", ["py", "o"]),
    ("でゃ", ["dy", "a"]), ("でゅ", ["dy", "u"]), ("でょ", ["dy", "o"]),
    # Digraphs
    ("つぁ", ["ts", "a"]), ("つぃ", ["ts", "i"]), ("つぇ", ["ts", "e"]), ("つぉ", ["ts", "o"]),
    ("ふぁ", ["f", "a"]),  ("ふぃ", ["f", "i"]),  ("ふぇ", ["f", "e"]),  ("ふぉ", ["f", "o"]),
    ("てぃ", ["t", "i"]),  ("でぃ", ["d", "i"]),
    # Basic kana
    ("か", ["k", "a"]), ("き", ["k", "i"]), ("く", ["k", "u"]), ("け", ["k", "e"]), ("こ", ["k", "o"]),
    ("さ", ["s", "a"]), ("し", ["sh", "i"]), ("す", ["s", "u"]), ("せ", ["s", "e"]), ("そ", ["s", "o"]),
    ("た", ["t", "a"]), ("ち", ["ch", "i"]), ("つ", ["ts", "u"]), ("て", ["t", "e"]), ("と", ["t", "o"]),
    ("な", ["n", "a"]), ("に", ["n", "i"]), ("ぬ", ["n", "u"]), ("ね", ["n", "e"]), ("の", ["n", "o"]),
    ("は", ["h", "a"]), ("ひ", ["h", "i"]), ("ふ", ["f", "u"]), ("へ", ["h", "e"]), ("ほ", ["h", "o"]),
    ("ま", ["m", "a"]), ("み", ["m", "i"]), ("む", ["m", "u"]), ("め", ["m", "e"]), ("も", ["m", "o"]),
    ("や", ["y", "a"]), ("ゆ", ["y", "u"]), ("よ", ["y", "o"]),
    ("ら", ["r", "a"]), ("り", ["r", "i"]), ("る", ["r", "u"]), ("れ", ["r", "e"]), ("ろ", ["r", "o"]),
    ("わ", ["w", "a"]), ("ゐ", ["w", "i"]), ("ゑ", ["w", "e"]), ("を", ["w", "o"]),
    # Voiced
    ("が", ["g", "a"]), ("ぎ", ["g", "i"]), ("ぐ", ["g", "u"]), ("げ", ["g", "e"]), ("ご", ["g", "o"]),
    ("ざ", ["z", "a"]), ("じ", ["j", "i"]), ("ず", ["z", "u"]), ("ぜ", ["z", "e"]), ("ぞ", ["z", "o"]),
    ("だ", ["d", "a"]), ("ぢ", ["d", "i"]), ("づ", ["z", "u"]), ("で", ["d", "e"]), ("ど", ["d", "o"]),
    ("ば", ["b", "a"]), ("び", ["b", "i"]), ("ぶ", ["b", "u"]), ("べ", ["b", "e"]), ("ぼ", ["b", "o"]),
    ("ぱ", ["p", "a"]), ("ぴ", ["p", "i"]), ("ぷ", ["p", "u"]), ("ぺ", ["p", "e"]), ("ぽ", ["p", "o"]),
    # Vowels
    ("あ", ["a"]), ("い", ["i"]), ("う", ["u"]), ("え", ["e"]), ("お", ["o"]),
    # Long vowel mark
    ("ー", []),  # handled by repeating previous vowel
]

# Build katakana table from hiragana
_KATA_OFFSET = ord("ァ") - ord("ぁ")
_KANA_MAP: list[tuple[str, list[str]]] = list(_KANA_TABLE)
for _kana, _phs in _KANA_TABLE:
    _kata = "".join(
        chr(ord(c) + _KATA_OFFSET) if "ぁ" <= c <= "ん" else c for c in _kana
    )
    if _kata != _kana:
        _KANA_MAP.append((_kata, _phs))

# Sort by length descending for greedy matching
_KANA_MAP.sort(key=lambda x: len(x[0]), reverse=True)


def _english_to_kana(word: str) -> str:
    """Approximate English word → katakana using simple phonetic rules."""
    word = word.lower().strip()
    if not word:
        return ""

    # Common English word → katakana dictionary
    _WORD_MAP = {
        "love": "ラブ", "story": "ストーリー", "forever": "フォーエバー",
        "hello": "ハロー", "world": "ワールド", "baby": "ベイビー",
        "dream": "ドリーム", "heart": "ハート", "star": "スター",
        "night": "ナイト", "light": "ライト", "time": "タイム",
        "life": "ライフ", "day": "デイ", "way": "ウェイ",
        "you": "ユー", "me": "ミー", "my": "マイ", "your": "ユア",
        "the": "ザ", "and": "アンド", "for": "フォー", "with": "ウィズ",
        "this": "ディス", "that": "ザット",
        "i": "アイ", "a": "ア",
        "oh": "オー", "yeah": "イェア", "no": "ノー", "yes": "イエス",
        "one": "ワン", "two": "ツー", "three": "スリー",
        "sing": "シング", "song": "ソング", "music": "ミュージック",
        "kiss": "キス", "smile": "スマイル", "cry": "クライ",
        "fly": "フライ", "sky": "スカイ", "rain": "レイン",
        "fire": "ファイア", "water": "ウォーター",
        "true": "トゥルー", "blue": "ブルー", "new": "ニュー",
        "good": "グッド", "bad": "バッド",
        "happy": "ハッピー", "crazy": "クレイジー",
        "sweet": "スウィート", "bitter": "ビター",
        "angel": "エンジェル", "devil": "デビル",
        "boy": "ボーイ", "girl": "ガール",
        "morning": "モーニング", "evening": "イブニング",
        "summer": "サマー", "winter": "ウィンター",
        "spring": "スプリング", "fall": "フォール",
        "color": "カラー", "rainbow": "レインボー",
        "feeling": "フィーリング", "everything": "エブリシング",
        "nothing": "ナッシング", "something": "サムシング",
    }

    if word in _WORD_MAP:
        return _WORD_MAP[word]

    # Fallback: rough letter-by-letter approximation
    _LETTER_MAP = {
        "a": "エイ", "b": "ビー", "c": "シー", "d": "ディー", "e": "イー",
        "f": "エフ", "g": "ジー", "h": "エイチ", "i": "アイ", "j": "ジェイ",
        "k": "ケイ", "l": "エル", "m": "エム", "n": "エヌ", "o": "オー",
        "p": "ピー", "q": "キュー", "r": "アール", "s": "エス", "t": "ティー",
        "u": "ユー", "v": "ブイ", "w": "ダブリュー", "x": "エックス",
        "y": "ワイ", "z": "ゼット",
    }
    # For unknown words, just do letter-by-letter (better than nothing)
    return "".join(_LETTER_MAP.get(c, "") for c in word)


def _number_to_kana(num_str: str) -> str:
    """Convert a numeric string to Japanese reading in hiragana."""
    _DIGITS = {
        "0": "ぜろ", "1": "いち", "2": "に", "3": "さん", "4": "よん",
        "5": "ご", "6": "ろく", "7": "なな", "8": "はち", "9": "きゅう",
    }
    _PLACES = {
        1: "", 2: "じゅう", 3: "ひゃく", 4: "せん",
    }
    _BIG = {
        4: "まん", 8: "おく",
    }

    num_str = num_str.strip()
    if not num_str.isdigit():
        return "".join(_DIGITS.get(c, "") for c in num_str)

    n = int(num_str)
    if n == 0:
        return "ぜろ"
    if n < 0:
        return "まいなす" + _number_to_kana(str(-n))

    # Simple approach: read each digit for large numbers
    if n >= 100000000:
        return "".join(_DIGITS.get(c, "") for c in str(n))

    # Handle up to 99,999,999
    parts = []

    if n >= 10000:
        man = n // 10000
        n %= 10000
        if man > 1:
            parts.append(_number_to_kana(str(man)))
        parts.append("まん")

    if n >= 1000:
        sen = n // 1000
        n %= 1000
        if sen == 3:
            parts.append("さんぜん")
        elif sen > 1:
            parts.append(_number_to_kana(str(sen)) + "せん")
        else:
            parts.append("せん")

    if n >= 100:
        hyaku = n // 100
        n %= 100
        if hyaku == 3:
            parts.append("さんびゃく")
        elif hyaku == 6:
            parts.append("ろっぴゃく")
        elif hyaku == 8:
            parts.append("はっぴゃく")
        elif hyaku > 1:
            parts.append(_DIGITS[str(hyaku)] + "ひゃく")
        else:
            parts.append("ひゃく")

    if n >= 10:
        juu = n // 10
        n %= 10
        if juu > 1:
            parts.append(_DIGITS[str(juu)] + "じゅう")
        else:
            parts.append("じゅう")

    if n > 0:
        parts.append(_DIGITS[str(n)])

    return "".join(parts)


def _kanji_to_kana(text: str) -> str:
    """Convert kanji/English/number text to hiragana using pykakasi.

    English words are converted to katakana approximations.
    Numbers are converted to Japanese readings.
    """
    from pykakasi import kakasi

    kks = kakasi()
    result = []
    for item in kks.convert(text):
        orig = item["orig"].strip()
        hira = item["hira"]

        # pykakasi leaves numbers unchanged — convert them
        if orig and orig.isdigit():
            result.append(_number_to_kana(orig))
        # pykakasi leaves English unchanged — convert it ourselves
        elif orig and hira == orig and any(c.isascii() and c.isalpha() for c in orig):
            words = orig.split()
            for w in words:
                kana = _english_to_kana(w)
                if kana:
                    result.append(kana)
        else:
            result.append(hira)
    return "".join(result)


def _kana_to_phonemes(text: str) -> list[str]:
    """Convert kana text to phoneme list using built-in table."""
    result: list[str] = []
    i = 0
    last_vowel = "a"

    while i < len(text):
        matched = False
        for kana, phs in _KANA_MAP:
            if text[i:].startswith(kana):
                if kana in ("ー", "ー"):
                    result.append(last_vowel)
                else:
                    result.extend(phs)
                    for p in reversed(phs):
                        if p in ("a", "i", "u", "e", "o"):
                            last_vowel = p
                            break
                i += len(kana)
                matched = True
                break
        if not matched:
            i += 1

    return result


def _has_non_kana(text: str) -> bool:
    """Check if text contains kanji or other non-kana characters that need conversion."""
    for ch in text:
        if ch in " \t\n、。,.!?！？…―ー・「」『』（）()~～":
            continue
        # Hiragana range: U+3040-U+309F, Katakana range: U+30A0-U+30FF
        if "\u3040" <= ch <= "\u309F" or "\u30A0" <= ch <= "\u30FF":
            continue
        # If we find a character outside kana ranges, it's likely kanji/romaji
        if "\u4E00" <= ch <= "\u9FFF" or "\u3400" <= ch <= "\u4DBF":
            return True  # CJK kanji
        if ch.isascii() and ch.isalpha():
            return True  # Romaji
    return False


class JapaneseG2P(G2PBase):
    def __init__(self):
        self._jtalk = None
        self._has_kakasi = False

        # Try pyopenjtalk first (best quality)
        try:
            import pyopenjtalk
            self._jtalk = pyopenjtalk
            return
        except ImportError:
            pass

        # Try pykakasi for kanji support
        try:
            import pykakasi
            self._has_kakasi = True
        except ImportError:
            pass

    def convert(self, text: str) -> list[str]:
        """Convert Japanese text (kanji/hiragana/katakana) to phoneme sequence."""
        if self._jtalk is not None:
            return self._convert_jtalk(text)
        return self._convert_builtin(text)

    def _convert_jtalk(self, text: str) -> list[str]:
        phoneme_str = self._jtalk.g2p(text, kana=False)
        raw_phonemes = phoneme_str.split()
        result = ["SP"]
        for ph in raw_phonemes:
            mapped = _JTALK_TO_DS.get(ph, ph)
            if mapped:
                result.append(mapped)
        result.append("SP")
        return result

    def _convert_builtin(self, text: str) -> list[str]:
        """Built-in conversion: kanji→kana (via pykakasi) → phonemes."""
        # Convert kanji to kana if needed
        if _has_non_kana(text) and self._has_kakasi:
            text = _kanji_to_kana(text)
        elif _has_non_kana(text):
            # No converter available — warn but try anyway
            pass

        phonemes = _kana_to_phonemes(text)
        if not phonemes:
            return ["SP"]
        return ["SP"] + phonemes + ["SP"]

    def get_phoneme_set(self) -> set[str]:
        return JA_PHONEMES.copy()

    def get_language_code(self) -> str:
        return "ja"
