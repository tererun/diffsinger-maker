"""Base G2P (Grapheme-to-Phoneme) interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class G2PBase(ABC):
    """Base class for grapheme-to-phoneme converters."""

    @abstractmethod
    def convert(self, text: str) -> list[str]:
        """Convert text to a list of phonemes."""
        ...

    @abstractmethod
    def get_phoneme_set(self) -> set[str]:
        """Return the set of all phonemes this G2P can produce."""
        ...

    @abstractmethod
    def get_language_code(self) -> str:
        """Return ISO 639 language code."""
        ...


def get_g2p(language: str) -> G2PBase:
    """Factory: return a G2P converter for the given language."""
    if language == "ja":
        from dsmaker.g2p.japanese import JapaneseG2P
        return JapaneseG2P()
    elif language == "en":
        from dsmaker.g2p.english import EnglishG2P
        return EnglishG2P()
    elif language == "zh":
        from dsmaker.g2p.mandarin import MandarinG2P
        return MandarinG2P()
    else:
        raise ValueError(f"Unsupported language: {language}")
