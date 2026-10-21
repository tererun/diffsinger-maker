"""Step 3: SOFA forced alignment — align phonemes to audio."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rich.console import Console

from dsmaker.config import ProjectConfig
from dsmaker.utils.download import ensure_sofa, get_sofa_model_path, get_cache_dir

console = Console()


def _find_sofa_dictionary(sofa_dir: Path, language: str) -> Path | None:
    """Find the best matching SOFA dictionary for the language.

    Search order:
    1. Downloaded model directory (~/.cache/dsmaker/sofa_models/)
    2. SOFA repo dictionary/ folder
    """
    # Check downloaded model directory first (model releases include dictionaries)
    models_dir = get_cache_dir() / "sofa_models"
    if models_dir.exists():
        for f in models_dir.glob("*.txt"):
            if language in f.stem.lower() or "japanese" in f.stem.lower() and language == "ja":
                return f
            if "english" in f.stem.lower() and language == "en":
                return f

    # SOFA repo dictionary/ folder
    dict_dir = sofa_dir / "dictionary"
    if dict_dir.exists():
        # Language-specific patterns
        patterns = {
            "ja": ["japanese", "ja"],
            "zh": ["opencpop", "mandarin", "zh"],
            "en": ["english", "en"],
        }
        for pattern in patterns.get(language, [language]):
            for f in dict_dir.iterdir():
                if pattern in f.stem.lower():
                    return f

        # Last resort: any txt
        dicts = list(dict_dir.glob("*.txt"))
        if dicts:
            return dicts[0]

    return None


def _convert_labs_to_syllable_format(segments_dir: Path, language: str) -> None:
    """Convert phoneme-level .lab files to syllable-level for SOFA.

    SOFA expects syllable tokens (e.g., 'ka shi') that it maps through
    its dictionary to phonemes. Our G2P writes phoneme tokens (e.g., 'k a sh i').
    This converts phoneme sequences back to syllables.
    """
    # Japanese: group consonant+vowel pairs into syllables
    vowels = {"a", "i", "u", "e", "o"}
    special = {"SP", "AP", "N", "cl"}

    for lab_path in sorted(segments_dir.glob("*.lab")):
        content = lab_path.read_text(encoding="utf-8").strip()
        phonemes = content.split()

        if not phonemes:
            continue

        syllables = []
        i = 0
        while i < len(phonemes):
            ph = phonemes[i]

            if ph in special:
                syllables.append(ph)
                i += 1
            elif ph in vowels:
                # Standalone vowel
                syllables.append(ph)
                i += 1
            else:
                # Consonant — check if next is a vowel to form syllable
                if i + 1 < len(phonemes) and phonemes[i + 1] in vowels:
                    syllables.append(ph + phonemes[i + 1])
                    i += 2
                else:
                    # Consonant cluster or final consonant
                    syllables.append(ph)
                    i += 1

        lab_path.write_text(" ".join(syllables), encoding="utf-8")


def run_label(project_dir: Path, config: ProjectConfig) -> list[Path]:
    """Run SOFA forced alignment on all segments."""
    segments_dir = project_dir / "segments"

    lab_files = sorted(segments_dir.glob("*.lab"))
    wav_files = sorted(segments_dir.glob("*.wav"))
    if not lab_files:
        console.print("[red]No .lab files found. Run 'dsmaker transcribe' first.[/red]")
        return []
    if not wav_files:
        console.print("[red]No .wav files found in segments/[/red]")
        return []

    console.print("[bold]Setting up SOFA forced aligner...[/bold]")

    sofa_dir = ensure_sofa()

    try:
        model_path = get_sofa_model_path(config.language)
    except FileNotFoundError:
        return []

    # Convert .lab files to syllable format for SOFA
    console.print("  Converting .lab files to syllable format...")
    _convert_labs_to_syllable_format(segments_dir, config.language)

    # Find dictionary
    dictionary = _find_sofa_dictionary(sofa_dir, config.language)
    dict_arg = []
    if dictionary:
        dict_arg = ["--dictionary", str(dictionary)]
        console.print(f"  Using dictionary: {dictionary.name}")

    console.print(f"  Using model: {model_path.name}")
    console.print(f"  Processing {len(wav_files)} segments...")

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
        for subdir in segments_dir.iterdir():
            if subdir.is_dir():
                textgrid_files = sorted(subdir.glob("*.TextGrid"))
                if textgrid_files:
                    for tg in textgrid_files:
                        tg.rename(segments_dir / tg.name)
                    textgrid_files = sorted(segments_dir.glob("*.TextGrid"))
                    break

    console.print(
        f"[green]SOFA alignment complete: {len(textgrid_files)} TextGrid files[/green]"
    )
    return textgrid_files
