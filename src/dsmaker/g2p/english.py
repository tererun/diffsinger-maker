"""English G2P using g2p_en."""

from __future__ import annotations

from dsmaker.g2p.base import G2PBase

EN_PHONEMES = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH",
    "EH", "ER", "EY",
    "F", "G", "HH",
    "IH", "IY",
    "JH", "K", "L", "M", "N", "NG",
    "OW", "OY",
    "P", "R", "S", "SH",
    "T", "TH",
    "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
    "SP", "AP",
}


class EnglishG2P(G2PBase):
    def __init__(self):
        try:
            from g2p_en import G2p
            self._g2p = G2p()
        except ImportError:
            raise ImportError(
                "g2p_en is required for English G2P. "
                "Install with: pip install 'diffsinger-maker[en]'"
            )

    def convert(self, text: str) -> list[str]:
        raw = self._g2p(text)
        result = ["SP"]
        for ph in raw:
            # g2p_en returns phonemes with optional stress numbers; strip them
            cleaned = ph.strip()
            if not cleaned or cleaned == " ":
                continue
            # Remove stress markers (0,1,2)
            base = "".join(c for c in cleaned if not c.isdigit())
            if base in EN_PHONEMES:
                result.append(base)
        result.append("SP")
        return result

    def get_phoneme_set(self) -> set[str]:
        return EN_PHONEMES.copy()

    def get_language_code(self) -> str:
        return "en"
