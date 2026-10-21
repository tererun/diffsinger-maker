"""CLI entry point for dsmaker."""

from __future__ import annotations

import click
from pathlib import Path
from rich.console import Console

from dsmaker import __version__
from dsmaker.config import ProjectConfig, CONFIG_FILENAME

console = Console()


def _find_project_dir() -> Path:
    """Find project directory by looking for dsmaker.yaml in cwd or parents."""
    cwd = Path.cwd()
    if (cwd / CONFIG_FILENAME).exists():
        return cwd
    for parent in cwd.parents:
        if (parent / CONFIG_FILENAME).exists():
            return parent
    return cwd


def _load_config(project_dir: Path | None = None) -> tuple[Path, ProjectConfig]:
    if project_dir is None:
        project_dir = _find_project_dir()
    else:
        project_dir = Path(project_dir)
    config = ProjectConfig.load(project_dir)
    return project_dir, config


@click.group()
@click.version_option(version=__version__)
def main():
    """dsmaker: DiffSinger voice model builder for OpenUtau."""
    pass


@main.command()
@click.argument("name")
@click.option("--lang", default="ja", help="Language (ja/en/zh)")
@click.option("--dir", "directory", default=None, help="Project directory (default: ./<name>)")
def init(name: str, lang: str, directory: str | None):
    """Initialize a new voice model project."""
    project_dir = Path(directory) if directory else Path.cwd() / name
    project_dir.mkdir(parents=True, exist_ok=True)

    config = ProjectConfig(name=name, language=lang)

    # Create directory structure
    (project_dir / "raw" / "songs").mkdir(parents=True, exist_ok=True)
    (project_dir / "raw" / "lyrics").mkdir(parents=True, exist_ok=True)
    (project_dir / "segments").mkdir(exist_ok=True)
    (project_dir / "dataset" / "raw" / "wavs").mkdir(parents=True, exist_ok=True)
    (project_dir / "training").mkdir(exist_ok=True)
    (project_dir / "export").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)

    config_path = config.save(project_dir)
    console.print(f"[green]Project initialized:[/green] {project_dir}")
    console.print(f"  Config: {config_path}")
    console.print()
    console.print("Next steps:")
    console.print(f"  1. Place your WAV recordings in [bold]{project_dir / 'raw' / 'songs'}[/bold]")
    console.print(f"  2. Place lyrics text files in [bold]{project_dir / 'raw' / 'lyrics'}[/bold]")
    console.print(f"  3. Run [bold]dsmaker slice[/bold] to split audio into segments")


@main.command()
@click.option("--project", default=None, help="Project directory")
def slice(project: str | None):
    """Slice audio files into segments."""
    from dsmaker.steps.slice import run_slice

    project_dir, config = _load_config(project)
    run_slice(project_dir, config)


@main.command()
@click.option("--project", default=None, help="Project directory")
def transcribe(project: str | None):
    """Convert lyrics to phoneme sequences."""
    from dsmaker.steps.transcribe import run_transcribe

    project_dir, config = _load_config(project)
    run_transcribe(project_dir, config)


@main.command()
@click.option("--project", default=None, help="Project directory")
def label(project: str | None):
    """Run SOFA forced alignment on segments."""
    from dsmaker.steps.label import run_label

    project_dir, config = _load_config(project)
    run_label(project_dir, config)


@main.command()
@click.option("--project", default=None, help="Project directory")
def prepare(project: str | None):
    """Generate DiffSinger training dataset."""
    from dsmaker.steps.prepare import run_prepare

    project_dir, config = _load_config(project)
    run_prepare(project_dir, config)


@main.command()
@click.option("--project", default=None, help="Project directory")
@click.option("--target", type=click.Choice(["acoustic", "variance", "both"]), default="both")
def train(project: str | None, target: str):
    """Train DiffSinger models."""
    from dsmaker.steps.train import run_train

    project_dir, config = _load_config(project)
    run_train(project_dir, config, target)


@main.command("export")
@click.option("--project", default=None, help="Project directory")
def export_cmd(project: str | None):
    """Export trained models to ONNX."""
    from dsmaker.steps.export import run_export

    project_dir, config = _load_config(project)
    run_export(project_dir, config)


@main.command()
@click.option("--project", default=None, help="Project directory")
@click.option("--zip", "make_zip", is_flag=True, help="Also create a zip file")
def package(project: str | None, make_zip: bool):
    """Package models for OpenUtau."""
    from dsmaker.steps.package import run_package

    project_dir, config = _load_config(project)
    run_package(project_dir, config, make_zip)


@main.command()
@click.option("--audio", required=True, type=click.Path(exists=True), help="Audio files directory")
@click.option("--lyrics", required=True, type=click.Path(exists=True), help="Lyrics files directory")
@click.option("--name", required=True, help="Singer name")
@click.option("--lang", default="ja", help="Language (ja/en/zh)")
@click.option("--interactive/--no-interactive", default=True, help="Interactive mode with confirmations")
@click.option("--out", default=None, help="Output directory")
@click.option("--backend", type=click.Choice(["local", "colab"]), default="local", help="Training backend")
def make(audio: str, lyrics: str, name: str, lang: str, interactive: bool, out: str | None, backend: str):
    """Build a complete DiffSinger voice model (one-command pipeline)."""
    from dsmaker.pipeline import run_pipeline

    run_pipeline(
        audio_dir=Path(audio),
        lyrics_dir=Path(lyrics),
        name=name,
        lang=lang,
        interactive=interactive,
        output_dir=Path(out) if out else None,
        backend=backend,
    )


@main.command("colab-export")
@click.option("--project", default=None, help="Project directory")
def colab_export(project: str | None):
    """Export dataset as zip for Google Colab training."""
    from dsmaker.steps.train import export_for_colab

    project_dir, config = _load_config(project)
    export_for_colab(project_dir, config)


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=7860, type=int, help="Port number")
@click.option("--share", is_flag=True, help="Create a public Gradio link")
def ui(host: str, port: int, share: bool):
    """Launch the web UI (Gradio)."""
    from dsmaker.web import launch_ui

    console.print(f"[bold]Starting dsmaker Web UI...[/bold]")
    console.print(f"  Open http://{host}:{port} in your browser")
    launch_ui(host=host, port=port, share=share)


if __name__ == "__main__":
    main()
