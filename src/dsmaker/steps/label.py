"""Step 3: SOFA forced alignment — align phonemes to audio."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rich.console import Console
from tqdm import tqdm

from dsmaker.config import ProjectConfig
from dsmaker.utils.download import ensure_sofa, get_sofa_model_path

console = Console()

# Language → SOFA dictionary filename mapping
_DICT_MAP = {
    "ja": "japanese.txt",
    "zh": "opencpop-extension.txt",
    "en": "english.txt",
}


def _find_sofa_dictionary(sofa_dir: Path, language: str) -> Path | None:
    """Find the best matching dictionary file in the SOFA repo."""
    dict_dir = sofa_dir / "dictionary"
    if not dict_dir.exists():
        return None

    # Try the known filename first
    known = _DICT_MAP.get(language)
    if known:
        candidate = dict_dir / known
        if candidate.exists():
            return candidate

    # Fallback: look for any file containing the language name
    for f in dict_dir.iterdir():
        if language in f.stem.lower():
            return f

    # Last resort: list all available dictionaries
    dicts = list(dict_dir.glob("*.txt"))
    if dicts:
        return dicts[0]

    return None


def run_label(project_dir: Path, config: ProjectConfig) -> list[Path]:
    """Run SOFA forced alignment on all segments."""
    segments_dir = project_dir / "segments"

    # Check that lab files exist
    lab_files = sorted(segments_dir.glob("*.lab"))
    wav_files = sorted(segments_dir.glob("*.wav"))
    if not lab_files:
        console.print("[red]No .lab files found. Run 'dsmaker transcribe' first.[/red]")
        return []
    if not wav_files:
        console.print("[red]No .wav files found in segments/[/red]")
        return []

    console.print("[bold]Setting up SOFA forced aligner...[/bold]")

    # Ensure SOFA is available
    sofa_dir = ensure_sofa()

    # Get model checkpoint
    try:
        model_path = get_sofa_model_path(config.language)
    except FileNotFoundError:
        return []

    # Find dictionary
    dictionary = _find_sofa_dictionary(sofa_dir, config.language)
    dict_arg = []
    if dictionary:
        dict_arg = ["--dictionary", str(dictionary)]
        console.print(f"  Using dictionary: {dictionary.name}")

    console.print(f"  Using model: {model_path.name}")
    console.print(f"  Processing {len(wav_files)} segments...")

    # Run SOFA inference
    # SOFA expects a folder with wav+lab pairs and produces TextGrid/htk output
    cmd = [
        sys.executable,
        str(sofa_dir / "infer.py"),
        "--ckpt", str(model_path),
        "--folder", str(segments_dir),
        "--in_format", "lab",
        "--out_formats", "TextGrid,htk",
    ] + dict_arg

    try:
        result = subprocess.run(
            cmd,
            cwd=str(sofa_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            console.print(f"[dim]{result.stdout.strip()}[/dim]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]SOFA failed:[/red]")
        if e.stderr:
            console.print(f"[red]{e.stderr}[/red]")
        raise

    # Collect output TextGrid files
    textgrid_files = sorted(segments_dir.glob("*.TextGrid"))
    if not textgrid_files:
        # SOFA may place TextGrids in a subdirectory
        for subdir in segments_dir.iterdir():
            if subdir.is_dir():
                textgrid_files = sorted(subdir.glob("*.TextGrid"))
                if textgrid_files:
                    # Move them to segments/ for easier access
                    for tg in textgrid_files:
                        tg.rename(segments_dir / tg.name)
                    textgrid_files = sorted(segments_dir.glob("*.TextGrid"))
                    break

    console.print(
        f"[green]SOFA alignment complete: {len(textgrid_files)} TextGrid files[/green]"
    )
    return textgrid_files
