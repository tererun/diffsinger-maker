"""Audio processing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio file. Resample to target sr if specified."""
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)

    # soundfile returns (samples, channels) — we want (channels, samples) or 1D
    if audio.ndim == 2:
        audio = audio.T  # (channels, samples)

    # Resample if needed
    if sr is not None and sr != sample_rate:
        try:
            import soxr
            if audio.ndim == 2:
                audio = np.array([soxr.resample(ch, sample_rate, sr) for ch in audio])
            else:
                audio = soxr.resample(audio, sample_rate, sr)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = sr / sample_rate
            if audio.ndim == 2:
                new_len = int(audio.shape[1] * ratio)
                indices = np.linspace(0, audio.shape[1] - 1, new_len)
                audio = np.array([np.interp(indices, np.arange(audio.shape[1]), ch) for ch in audio])
            else:
                new_len = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_len)
                audio = np.interp(indices, np.arange(len(audio)), audio)
        sample_rate = sr

    return audio.astype(np.float32), sample_rate


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    if len(audio.shape) > 1:
        audio = audio.T
    sf.write(str(path), audio, sr)


def get_audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.duration


def get_audio_files(directory: Path) -> list[Path]:
    extensions = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    files = []
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in extensions:
            files.append(f)
    return files
