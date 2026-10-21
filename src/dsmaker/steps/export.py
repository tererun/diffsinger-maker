"""Step 6: Export trained models to ONNX format."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rich.console import Console

from dsmaker.config import ProjectConfig
from dsmaker.utils.download import ensure_diffsinger
from dsmaker.utils.platform import ensure_dir

console = Console()


def run_export(project_dir: Path, config: ProjectConfig) -> dict[str, Path]:
    """Export acoustic and variance models to ONNX."""
    console.print("[bold]Exporting models to ONNX...[/bold]")

    ds_dir = ensure_diffsinger()
    export_dir = ensure_dir(project_dir / "export")
    exported: dict[str, Path] = {}

    # Export acoustic model
    acoustic_exp = f"{config.name}_acoustic"
    console.print(f"\n  Exporting acoustic model ({acoustic_exp})...")
    try:
        subprocess.run(
            [sys.executable, str(ds_dir / "scripts" / "export.py"),
             "acoustic", "--exp", acoustic_exp],
            cwd=str(ds_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        # Find the exported ONNX file
        ckpt_dir = ds_dir / "checkpoints" / acoustic_exp
        for onnx in ckpt_dir.glob("*.onnx"):
            dest = export_dir / onnx.name
            onnx.rename(dest) if not dest.exists() else None
            exported[onnx.stem] = dest
            console.print(f"    [green]{dest.name}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"  [red]Acoustic export failed: {e.stderr}[/red]")

    # Export variance model
    variance_exp = f"{config.name}_variance"
    console.print(f"\n  Exporting variance model ({variance_exp})...")
    try:
        subprocess.run(
            [sys.executable, str(ds_dir / "scripts" / "export.py"),
             "variance", "--exp", variance_exp],
            cwd=str(ds_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        ckpt_dir = ds_dir / "checkpoints" / variance_exp
        for onnx in ckpt_dir.glob("*.onnx"):
            dest = export_dir / onnx.name
            onnx.rename(dest) if not dest.exists() else None
            exported[onnx.stem] = dest
            console.print(f"    [green]{dest.name}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"  [red]Variance export failed: {e.stderr}[/red]")

    console.print(f"\n[green]Exported {len(exported)} ONNX file(s) to {export_dir}[/green]")
    return exported
