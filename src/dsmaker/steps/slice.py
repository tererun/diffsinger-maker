"""Step 1: Audio slicing — split recordings into segments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console
from tqdm import tqdm

from dsmaker.config import ProjectConfig
from dsmaker.utils.audio import get_audio_files, load_audio, save_audio
from dsmaker.utils.platform import ensure_dir

console = Console()


class AudioSlicer:
    """Port of openvpi/audio-slicer (slicer2) as a self-contained class."""

    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 10,
        max_sil_kept: int = 500,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError("Must satisfy: min_length >= min_interval >= hop_size")
        if not max_sil_kept >= hop_size:
            raise ValueError("Must satisfy: max_sil_kept >= hop_size")
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_rms(self, audio: np.ndarray) -> np.ndarray:
        length = int(np.ceil(len(audio) / self.hop_size))
        rms = np.zeros(length)
        for i in range(length):
            start = i * self.hop_size - self.win_size // 2
            end = start + self.win_size
            start = max(0, start)
            end = min(len(audio), end)
            rms[i] = np.sqrt(np.mean(audio[start:end] ** 2))
        return rms

    def slice(self, waveform: np.ndarray) -> list[np.ndarray]:
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform

        if len(samples) <= self.min_length * self.hop_size:
            return [waveform]

        rms = self._apply_rms(samples)
        sil_tags: list[tuple[int, int]] = []
        silence_start: int | None = None
        clip_start = 0

        for i, rms_val in enumerate(rms):
            if rms_val < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            is_leading = silence_start == 0 and i > self.max_sil_kept
            need_slice = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not is_leading and not need_slice:
                silence_start = None
                continue
            if i - silence_start <= self.max_sil_kept:
                pos = rms[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                mid = (silence_start + i) // 2
                pos1 = rms[silence_start : mid + 1].argmin() + silence_start
                pos2 = rms[mid : i + 1].argmin() + mid
                if silence_start == 0:
                    sil_tags.append((0, pos2))
                else:
                    sil_tags.append((pos1, pos2))
                clip_start = pos2
            else:
                pos1 = (
                    rms[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos2 = rms[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos1))
                else:
                    sil_tags.append((pos1, pos2))
                clip_start = pos2
            silence_start = None

        total_frames = rms.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames))

        if not sil_tags:
            return [waveform]

        chunks = []
        if sil_tags[0][0] > 0:
            chunks.append(self._get_chunk(waveform, 0, sil_tags[0][0]))
        for i in range(len(sil_tags) - 1):
            chunks.append(
                self._get_chunk(waveform, sil_tags[i][1], sil_tags[i + 1][0])
            )
        if sil_tags[-1][1] < total_frames:
            chunks.append(self._get_chunk(waveform, sil_tags[-1][1], total_frames))

        return chunks

    def _get_chunk(
        self, waveform: np.ndarray, start: int, end: int
    ) -> np.ndarray:
        s = start * self.hop_size
        e = min(end * self.hop_size, waveform.shape[-1])
        if len(waveform.shape) > 1:
            return waveform[:, s:e]
        return waveform[s:e]


def run_slice(project_dir: Path, config: ProjectConfig) -> list[Path]:
    audio_dir = project_dir / config.audio_dir
    segments_dir = ensure_dir(project_dir / "segments")

    audio_files = get_audio_files(audio_dir)
    if not audio_files:
        console.print(f"[red]No audio files found in {audio_dir}[/red]")
        return []

    console.print(f"[bold]Slicing {len(audio_files)} audio file(s)...[/bold]")

    output_files = []
    for audio_file in tqdm(audio_files, desc="Slicing"):
        audio, sr = load_audio(audio_file, sr=config.sample_rate)
        slicer = AudioSlicer(
            sr=sr,
            threshold=config.slice.threshold,
            min_length=config.slice.min_length,
            min_interval=config.slice.min_interval,
            hop_size=config.slice.hop_size,
            max_sil_kept=config.slice.max_sil_kept,
        )
        chunks = slicer.slice(audio)
        stem = audio_file.stem
        for i, chunk in enumerate(chunks):
            out_path = segments_dir / f"{stem}_{i:04d}.wav"
            save_audio(out_path, chunk, sr)
            output_files.append(out_path)

    console.print(
        f"[green]Created {len(output_files)} segments in {segments_dir}[/green]"
    )
    return output_files
