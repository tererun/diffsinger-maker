"""Step 2: Convert lyrics text to phoneme sequences and .lab files."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from dsmaker.config import ProjectConfig
from dsmaker.g2p.base import get_g2p

console = Console()


def _read_lyrics_file(path: Path) -> list[str]:
    """Read a lyrics file — one line per segment."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def get_song_segments(segments_dir: Path) -> dict[str, list[Path]]:
    """Group segment WAV files by song name.

    Returns dict: song_stem → [segment_path, ...]
    """
    song_segments: dict[str, list[Path]] = {}
    for seg in sorted(segments_dir.glob("*.wav")):
        parts = seg.stem.rsplit("_", 1)
        song_stem = parts[0] if len(parts) == 2 and parts[1].isdigit() else seg.stem
        song_segments.setdefault(song_stem, []).append(seg)
    return song_segments


def _match_lyrics_to_segments(
    lyrics_dir: Path, segments_dir: Path
) -> list[tuple[Path, str]]:
    """Match lyrics to segments — per-song: each txt is the full lyrics for one song.

    Strategy:
    - Group segments by song stem (song1_0000.wav → song1)
    - Match each song to its lyrics file (song1.txt)
    - Split lyrics lines across that song's segments
    - If there's a single lyrics file, use it for all songs
    """
    segment_files = sorted(segments_dir.glob("*.wav"))
    lyrics_files = sorted(lyrics_dir.glob("*.txt"))

    if not segment_files:
        raise FileNotFoundError(f"No segment .wav files found in {segments_dir}")

    song_segments = get_song_segments(segments_dir)
    pairs: list[tuple[Path, str]] = []

    if not lyrics_files:
        # No lyrics files — return empty lyrics for all segments
        for seg in segment_files:
            pairs.append((seg, ""))
        return pairs

    if len(lyrics_files) == 1:
        # Single lyrics file → distribute lines across all segments
        lines = _read_lyrics_file(lyrics_files[0])
        for i, seg in enumerate(segment_files):
            lyric = lines[i] if i < len(lines) else ""
            pairs.append((seg, lyric))
    else:
        # Multiple lyrics files → match by song name
        lyrics_by_stem = {f.stem: f for f in lyrics_files}
        for song_stem, segs in song_segments.items():
            lf = lyrics_by_stem.get(song_stem)
            if lf is None:
                console.print(
                    f"[yellow]Warning: No lyrics file for '{song_stem}', "
                    f"skipping {len(segs)} segments[/yellow]"
                )
                for seg in segs:
                    pairs.append((seg, ""))
                continue
            lines = _read_lyrics_file(lf)
            for i, seg in enumerate(segs):
                lyric = lines[i] if i < len(lines) else ""
                pairs.append((seg, lyric))

    return pairs


def build_segment_lyrics_map(
    segments_dir: Path, lyrics_dir: Path
) -> list[dict]:
    """Build a list of {segment, song, lyric} dicts for the UI editor.

    After slicing, this gives the user a per-segment view they can edit.
    """
    song_segments = get_song_segments(segments_dir)
    lyrics_files = sorted(lyrics_dir.glob("*.txt"))
    lyrics_by_stem = {f.stem: f for f in lyrics_files}

    rows: list[dict] = []
    for song_stem, segs in song_segments.items():
        lf = lyrics_by_stem.get(song_stem)
        lines = _read_lyrics_file(lf) if lf else []

        # If only one lyrics file exists and names don't match, try it
        if not lf and len(lyrics_files) == 1:
            lines = _read_lyrics_file(lyrics_files[0])

        for i, seg in enumerate(segs):
            rows.append({
                "segment": seg.stem,
                "song": song_stem,
                "lyric": lines[i] if i < len(lines) else "",
                "path": str(seg),
            })
    return rows


def run_transcribe(
    project_dir: Path,
    config: ProjectConfig,
    interactive: bool = False,
    segment_lyrics: list[tuple[str, str]] | None = None,
) -> list[Path]:
    """Convert lyrics to phoneme .lab files for each segment.

    Args:
        segment_lyrics: Optional list of (segment_stem, lyric_text) overrides
                        from the UI editor. If provided, uses these instead of
                        reading from lyrics files.
    """
    segments_dir = project_dir / "segments"
    g2p = get_g2p(config.language)

    if segment_lyrics:
        # Use explicit segment→lyrics mapping (from UI)
        pairs: list[tuple[Path, str]] = []
        for stem, lyric in segment_lyrics:
            wav = segments_dir / f"{stem}.wav"
            if wav.exists():
                pairs.append((wav, lyric))
    else:
        lyrics_dir = project_dir / config.lyrics_dir
        pairs = _match_lyrics_to_segments(lyrics_dir, segments_dir)

    console.print(f"[bold]Converting {len(pairs)} segments to phonemes ({config.language})...[/bold]")

    lab_files = []
    results: list[tuple[str, str, list[str]]] = []

    for seg_path, lyric in pairs:
        if not lyric:
            phonemes = ["SP"]
        else:
            phonemes = g2p.convert(lyric)
        results.append((seg_path.stem, lyric, phonemes))

        lab_path = seg_path.with_suffix(".lab")
        lab_path.write_text(" ".join(phonemes), encoding="utf-8")
        lab_files.append(lab_path)

    if interactive:
        table = Table(title="Phoneme Conversion Results")
        table.add_column("Segment", style="cyan")
        table.add_column("Lyrics", style="white")
        table.add_column("Phonemes", style="green")
        for name, lyric, phs in results[:20]:
            table.add_row(name, lyric, " ".join(phs))
        if len(results) > 20:
            table.add_row("...", f"({len(results) - 20} more)", "...")
        console.print(table)
        console.print()
        console.print(
            "[dim]Lab files written. Edit .lab files in segments/ if needed.[/dim]"
        )
    else:
        console.print(f"[green]Created {len(lab_files)} .lab files[/green]")

    return lab_files
