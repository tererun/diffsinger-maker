"""Step 2: Convert lyrics text to phoneme sequences and .lab files."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from dsmaker.config import ProjectConfig
from dsmaker.g2p.base import get_g2p
from dsmaker.utils.platform import ensure_dir

console = Console()


def _read_lyrics_file(path: Path) -> list[str]:
    """Read a lyrics file, one line per segment."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def _match_lyrics_to_segments(
    lyrics_dir: Path, segments_dir: Path
) -> list[tuple[Path, str]]:
    """Match lyrics lines to segment WAV files.

    Strategy:
    - If a .txt file has the same stem as a song (before slicing), its lines
      are mapped sequentially to the segments from that song.
    - If there's a single lyrics file, all lines map to all segments in order.
    """
    segment_files = sorted(segments_dir.glob("*.wav"))
    lyrics_files = sorted(lyrics_dir.glob("*.txt"))

    if not lyrics_files:
        raise FileNotFoundError(f"No lyrics .txt files found in {lyrics_dir}")
    if not segment_files:
        raise FileNotFoundError(f"No segment .wav files found in {segments_dir}")

    # Group segments by song stem (before the _NNNN suffix)
    song_segments: dict[str, list[Path]] = {}
    for seg in segment_files:
        # segment name format: songname_0001.wav
        parts = seg.stem.rsplit("_", 1)
        song_stem = parts[0] if len(parts) == 2 and parts[1].isdigit() else seg.stem
        song_segments.setdefault(song_stem, []).append(seg)

    pairs: list[tuple[Path, str]] = []

    if len(lyrics_files) == 1:
        # Single lyrics file → map lines to all segments in order
        lines = _read_lyrics_file(lyrics_files[0])
        for i, seg in enumerate(segment_files):
            lyric = lines[i] if i < len(lines) else ""
            pairs.append((seg, lyric))
    else:
        # Multiple lyrics files → match by filename stem
        lyrics_by_stem = {f.stem: f for f in lyrics_files}
        for song_stem, segs in song_segments.items():
            lf = lyrics_by_stem.get(song_stem)
            if lf is None:
                console.print(
                    f"[yellow]Warning: No lyrics file for song '{song_stem}', skipping {len(segs)} segments[/yellow]"
                )
                continue
            lines = _read_lyrics_file(lf)
            for i, seg in enumerate(segs):
                lyric = lines[i] if i < len(lines) else ""
                pairs.append((seg, lyric))

    return pairs


def run_transcribe(
    project_dir: Path,
    config: ProjectConfig,
    interactive: bool = False,
) -> list[Path]:
    """Convert lyrics to phoneme .lab files for each segment."""
    lyrics_dir = project_dir / config.lyrics_dir
    segments_dir = project_dir / "segments"

    pairs = _match_lyrics_to_segments(lyrics_dir, segments_dir)
    g2p = get_g2p(config.language)

    console.print(f"[bold]Converting {len(pairs)} segments to phonemes ({config.language})...[/bold]")

    lab_files = []
    results: list[tuple[str, str, list[str]]] = []

    for seg_path, lyric in pairs:
        if not lyric:
            phonemes = ["SP"]
        else:
            phonemes = g2p.convert(lyric)
        results.append((seg_path.stem, lyric, phonemes))

        # Write .lab file (space-separated phoneme sequence)
        lab_path = seg_path.with_suffix(".lab")
        lab_path.write_text(" ".join(phonemes), encoding="utf-8")
        lab_files.append(lab_path)

    # Show results for confirmation
    if interactive:
        table = Table(title="Phoneme Conversion Results")
        table.add_column("Segment", style="cyan")
        table.add_column("Lyrics", style="white")
        table.add_column("Phonemes", style="green")
        for name, lyric, phs in results[:20]:  # show first 20
            table.add_row(name, lyric, " ".join(phs))
        if len(results) > 20:
            table.add_row("...", f"({len(results) - 20} more)", "...")
        console.print(table)
        console.print()
        console.print(
            "[dim]Lab files written. You can manually edit .lab files in the segments/ "
            "directory before proceeding to labeling.[/dim]"
        )
    else:
        console.print(f"[green]Created {len(lab_files)} .lab files[/green]")

    return lab_files
