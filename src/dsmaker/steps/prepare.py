"""Step 4: Prepare DiffSinger training dataset from aligned segments."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from rich.console import Console

from dsmaker.config import ProjectConfig
from dsmaker.utils.platform import ensure_dir

console = Console()


def _parse_textgrid(path: Path) -> list[tuple[float, float, str]]:
    """Parse a Praat TextGrid file and return intervals from the first tier.

    Returns list of (start_time, end_time, label).
    """
    text = path.read_text(encoding="utf-8")
    intervals: list[tuple[float, float, str]] = []

    lines = text.strip().splitlines()
    i = 0
    in_intervals = False

    while i < len(lines):
        line = lines[i].strip()

        if '"IntervalTier"' in line:
            in_intervals = True
            i += 1
            continue

        if in_intervals and line.startswith("xmin"):
            xmin = float(line.split("=")[1].strip())
            i += 1
            xmax = float(lines[i].strip().split("=")[1].strip())
            i += 1
            label_line = lines[i].strip()
            # Extract text between quotes
            label = label_line.split('"')[1] if '"' in label_line else label_line.split("=")[1].strip()
            intervals.append((xmin, xmax, label))
            i += 1
            continue

        i += 1

    return intervals


def _parse_htk_lab(path: Path) -> list[tuple[float, float, str]]:
    """Parse an HTK .lab file. Format: start end label (times in 100ns units)."""
    intervals = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            start = int(parts[0]) / 1e7  # 100ns → seconds
            end = int(parts[1]) / 1e7
            label = parts[2]
            intervals.append((start, end, label))
    return intervals


def _intervals_to_ph_seq_dur(
    intervals: list[tuple[float, float, str]],
) -> tuple[str, str]:
    """Convert intervals to ph_seq and ph_dur strings for transcriptions.csv."""
    phonemes = []
    durations = []
    for start, end, label in intervals:
        if not label or label == "":
            continue
        phonemes.append(label)
        durations.append(f"{end - start:.6f}")
    return " ".join(phonemes), " ".join(durations)


def run_prepare(project_dir: Path, config: ProjectConfig) -> Path:
    """Build the DiffSinger dataset directory from aligned segments."""
    segments_dir = project_dir / "segments"
    dataset_dir = ensure_dir(project_dir / "dataset" / "raw")
    wavs_dir = ensure_dir(dataset_dir / "wavs")

    # Find alignment files (prefer TextGrid, fall back to HTK)
    textgrid_files = sorted(segments_dir.glob("*.TextGrid"))
    htk_files = sorted(segments_dir.glob("*.htk"))

    if textgrid_files:
        alignment_files = textgrid_files
        parse_fn = _parse_textgrid
        console.print(f"  Using {len(alignment_files)} TextGrid files")
    elif htk_files:
        alignment_files = htk_files
        parse_fn = _parse_htk_lab
        console.print(f"  Using {len(alignment_files)} HTK lab files")
    else:
        console.print("[red]No alignment files found. Run 'dsmaker label' first.[/red]")
        return dataset_dir

    console.print(f"[bold]Preparing DiffSinger dataset...[/bold]")

    rows: list[dict[str, str]] = []
    copied = 0

    for align_file in alignment_files:
        stem = align_file.stem
        wav_file = segments_dir / f"{stem}.wav"
        if not wav_file.exists():
            console.print(f"[yellow]Warning: {wav_file.name} not found, skipping[/yellow]")
            continue

        intervals = parse_fn(align_file)
        if not intervals:
            console.print(f"[yellow]Warning: No intervals in {align_file.name}, skipping[/yellow]")
            continue

        ph_seq, ph_dur = _intervals_to_ph_seq_dur(intervals)
        if not ph_seq:
            continue

        # Copy WAV to dataset
        dest_wav = wavs_dir / wav_file.name
        if not dest_wav.exists():
            shutil.copy2(wav_file, dest_wav)
        copied += 1

        rows.append({
            "name": stem,
            "ph_seq": ph_seq,
            "ph_dur": ph_dur,
        })

    # Write transcriptions.csv
    csv_path = dataset_dir / "transcriptions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        writer.writerows(rows)

    console.print(f"[green]Dataset ready:[/green]")
    console.print(f"  WAV files: {copied}")
    console.print(f"  Transcriptions: {csv_path}")
    console.print(f"  Entries: {len(rows)}")

    return dataset_dir
