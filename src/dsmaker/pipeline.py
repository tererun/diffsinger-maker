"""Full pipeline orchestrator — the 'dsmaker make' one-command flow."""

from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from dsmaker.config import ProjectConfig
from dsmaker.utils.platform import ensure_dir, has_cuda

console = Console()


def run_pipeline(
    audio_dir: Path,
    lyrics_dir: Path,
    name: str,
    lang: str = "ja",
    interactive: bool = True,
    output_dir: Path | None = None,
    backend: str = "local",
) -> Path:
    """Run the complete DiffSinger voice model creation pipeline."""
    # Determine project directory
    project_dir = output_dir or (Path.cwd() / name)
    project_dir = project_dir.resolve()

    console.print(Panel(
        f"[bold]dsmaker[/bold] — Building voice model: [cyan]{name}[/cyan]\n"
        f"  Language: {lang}\n"
        f"  Audio: {audio_dir}\n"
        f"  Lyrics: {lyrics_dir}\n"
        f"  Output: {project_dir}\n"
        f"  Backend: {backend}",
        title="DiffSinger Maker",
    ))

    # ── Step 0: Initialize project ──
    console.print("\n[bold]Step 0/7: Initializing project...[/bold]")
    ensure_dir(project_dir)
    config = ProjectConfig(
        name=name,
        language=lang,
        audio_dir="raw/songs",
        lyrics_dir="raw/lyrics",
    )
    config.training.backend = backend

    # Copy audio and lyrics into project
    songs_dir = ensure_dir(project_dir / "raw" / "songs")
    lyrics_dest = ensure_dir(project_dir / "raw" / "lyrics")
    ensure_dir(project_dir / "segments")
    ensure_dir(project_dir / "dataset" / "raw" / "wavs")
    ensure_dir(project_dir / "training")
    ensure_dir(project_dir / "export")
    ensure_dir(project_dir / "output")

    for f in audio_dir.iterdir():
        if f.is_file() and f.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
            dest = songs_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)

    for f in lyrics_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".txt":
            dest = lyrics_dest / f.name
            if not dest.exists():
                shutil.copy2(f, dest)

    config.save(project_dir)
    console.print("  [green]Project initialized[/green]")

    # ── Step 1: Slice audio ──
    console.print("\n[bold]Step 1/7: Slicing audio...[/bold]")
    from dsmaker.steps.slice import run_slice
    segments = run_slice(project_dir, config)

    if not segments:
        console.print("[red]No segments created. Check your audio files.[/red]")
        return project_dir

    if interactive:
        console.print(f"\n  Created {len(segments)} segments.")
        if not Confirm.ask("  Continue to phoneme conversion?", default=True):
            console.print("[yellow]Stopped. Resume with 'dsmaker transcribe'[/yellow]")
            return project_dir

    # ── Step 2: Transcribe lyrics → phonemes ──
    console.print("\n[bold]Step 2/7: Converting lyrics to phonemes...[/bold]")
    from dsmaker.steps.transcribe import run_transcribe
    lab_files = run_transcribe(project_dir, config, interactive=interactive)

    if interactive:
        console.print(
            "\n  [dim]Review .lab files in segments/ and edit if needed.[/dim]"
        )
        if not Confirm.ask("  Continue to SOFA alignment?", default=True):
            console.print("[yellow]Stopped. Resume with 'dsmaker label'[/yellow]")
            return project_dir

    # ── Step 3: SOFA forced alignment ──
    console.print("\n[bold]Step 3/7: Running SOFA alignment...[/bold]")
    from dsmaker.steps.label import run_label
    textgrids = run_label(project_dir, config)

    if not textgrids:
        console.print("[red]Alignment failed. Check SOFA setup.[/red]")
        return project_dir

    if interactive and not Confirm.ask("  Continue to dataset preparation?", default=True):
        console.print("[yellow]Stopped. Resume with 'dsmaker prepare'[/yellow]")
        return project_dir

    # ── Step 4: Prepare dataset ──
    console.print("\n[bold]Step 4/7: Preparing DiffSinger dataset...[/bold]")
    from dsmaker.steps.prepare import run_prepare
    run_prepare(project_dir, config)

    # ── Step 5: Train ──
    if backend == "colab":
        console.print("\n[bold]Step 5/7: Exporting for Google Colab...[/bold]")
        from dsmaker.steps.train import export_for_colab
        export_for_colab(project_dir, config)
        console.print(
            "\n[bold yellow]Colab training:[/bold yellow]\n"
            f"  1. Open the notebook in export/{name}_train.ipynb with Google Colab\n"
            f"  2. Upload export/{name}_colab_data.zip when prompted\n"
            "  3. Run all cells\n"
            "  4. Download the exported ONNX zip\n"
            f"  5. Extract ONNX files to {project_dir / 'export'}\n"
            "  6. Run 'dsmaker package' to create the OpenUtau voice bank"
        )
        return project_dir

    console.print("\n[bold]Step 5/7: Training models...[/bold]")
    if interactive:
        if not has_cuda():
            console.print("[yellow]No CUDA GPU detected. Training on CPU will be very slow.[/yellow]")
        if not Confirm.ask("  Start training? (This may take hours)", default=True):
            console.print("[yellow]Stopped. Resume with 'dsmaker train'[/yellow]")
            return project_dir

    from dsmaker.steps.train import run_train
    run_train(project_dir, config, target="both")

    # ── Step 6: Export ONNX ──
    console.print("\n[bold]Step 6/7: Exporting to ONNX...[/bold]")
    from dsmaker.steps.export import run_export
    run_export(project_dir, config)

    # ── Step 7: Package ──
    console.print("\n[bold]Step 7/7: Packaging for OpenUtau...[/bold]")
    from dsmaker.steps.package import run_package
    output = run_package(project_dir, config, make_zip=True)

    console.print(Panel(
        f"[bold green]Voice model complete![/bold green]\n\n"
        f"  Output: {output}\n"
        f"  Copy to OpenUtau's Singers folder to use.",
        title="Done",
    ))

    return project_dir
