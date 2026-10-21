"""Mandarin Chinese G2P using pypinyin."""

from __future__ import annotations

from dsmaker.g2p.base import G2PBase

# DiffSinger Mandarin phoneme set (based on opencpop)
ZH_PHONEMES = {
    # initials
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "zh", "ch", "sh", "r",
    "z", "c", "s",
    "y", "w",
    # finals
    "a", "o", "e", "i", "u", "v",
    "ai", "ei", "ao", "ou",
    "an", "en", "ang", "eng", "ong",
    "er",
    "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
    "ua", "uo", "uai", "ui", "uan", "un", "uang",
    "ve", "van", "vn",
    # special
    "SP", "AP",
}

# Pinyin initial → phoneme mapping
_INITIALS = [
    "zh", "ch", "sh",  # must check longer ones first
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "r", "z", "c", "s",
    "y", "w",
]


def _split_pinyin(pinyin: str) -> tuple[str, str]:
    """Split pinyin into initial and final."""
    pinyin = pinyin.lower().strip()
    for ini in _INITIALS:
        if pinyin.startswith(ini):
            return ini, pinyin[len(ini):]
    return "", pinyin


class MandarinG2P(G2PBase):
    def __init__(self):
        try:
            import pypinyin
            self._pypinyin = pypinyin
        except ImportError:
            raise ImportError(
                "pypinyin is required for Mandarin G2P. "
                "Install with: pip install 'diffsinger-maker[zh]'"
            )

    def convert(self, text: str) -> list[str]:
        from pypinyin import pinyin, Style

        syllables = pinyin(text, style=Style.NORMAL, heteronym=False)
        result = ["SP"]
        for syl_list in syllables:
            syl = syl_list[0]
            if not syl.isalpha():
                continue
            initial, final = _split_pinyin(syl)
            if initial:
                result.append(initial)
            if final:
                result.append(final)
        result.append("SP")
        return result

    def get_phoneme_set(self) -> set[str]:
        return ZH_PHONEMES.copy()

    def get_language_code(self) -> str:
        return "zh"
