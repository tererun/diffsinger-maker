"""Step 5: Train DiffSinger models (local GPU or Colab export)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml
from rich.console import Console

from dsmaker.config import ProjectConfig
from dsmaker.utils.download import ensure_diffsinger
from dsmaker.utils.platform import ensure_dir, has_cuda

console = Console()


def _generate_acoustic_config(
    project_dir: Path, config: ProjectConfig, ds_dir: Path
) -> Path:
    """Generate DiffSinger acoustic training config YAML."""
    dataset_dir = project_dir / "dataset"
    raw_dir = dataset_dir / "raw"
    binary_dir = ensure_dir(dataset_dir / "binary")
    training_dir = ensure_dir(project_dir / "training" / "acoustic")

    cfg = {
        "base_config": [
            str(ds_dir / "configs" / "acoustic.yaml"),
        ],
        "datasets": [
            {
                "raw_data_dir": str(raw_dir),
                "speaker": config.name,
                "spk_id": 0,
                "language": config.language,
                "test_prefixes": [],
            }
        ],
        "binary_data_dir": str(binary_dir),
        "dictionary": f"dictionaries/{config.language}.txt",
        "num_spk": 1,
        "audio_sample_rate": config.sample_rate,
        "audio_num_mel_bins": 128,
        "hop_size": 512,
        "fft_size": 2048,
        "win_size": 2048,
        "max_batch_size": config.training.acoustic.batch_size,
        "max_updates": config.training.acoustic.max_updates,
        "accumulate_grad_batches": 1,
        "val_check_interval": 2000,
        "num_valid_plots": 4,
        "backbone_type": "lynxnet",
        "diffusion_type": "reflow",
        "use_spk_id": False,
    }

    # Pick test prefixes from first few entries
    csv_path = raw_dir / "transcriptions.csv"
    if csv_path.exists():
        import csv as csv_mod
        with open(csv_path, encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            names = [row["name"] for row in reader]
        if names:
            cfg["datasets"][0]["test_prefixes"] = names[:3]

    config_path = training_dir / "acoustic_config.yaml"
    config_path.write_text(
        yaml.dump(cfg, default_flow_style=False, allow_unicode=True)
    )
    return config_path


def _generate_variance_config(
    project_dir: Path, config: ProjectConfig, ds_dir: Path
) -> Path:
    """Generate DiffSinger variance training config YAML."""
    dataset_dir = project_dir / "dataset"
    raw_dir = dataset_dir / "raw"
    binary_dir = ensure_dir(dataset_dir / "binary_var")
    training_dir = ensure_dir(project_dir / "training" / "variance")

    cfg = {
        "base_config": [
            str(ds_dir / "configs" / "variance.yaml"),
        ],
        "datasets": [
            {
                "raw_data_dir": str(raw_dir),
                "speaker": config.name,
                "spk_id": 0,
                "language": config.language,
                "test_prefixes": [],
            }
        ],
        "binary_data_dir": str(binary_dir),
        "dictionary": f"dictionaries/{config.language}.txt",
        "num_spk": 1,
        "audio_sample_rate": config.sample_rate,
        "max_batch_size": config.training.variance.batch_size,
        "max_updates": config.training.variance.max_updates,
        "predict_dur": True,
        "predict_pitch": True,
        "predict_voicing": True,
        "predict_tension": True,
    }

    csv_path = raw_dir / "transcriptions.csv"
    if csv_path.exists():
        import csv as csv_mod
        with open(csv_path, encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            names = [row["name"] for row in reader]
        if names:
            cfg["datasets"][0]["test_prefixes"] = names[:3]

    config_path = training_dir / "variance_config.yaml"
    config_path.write_text(
        yaml.dump(cfg, default_flow_style=False, allow_unicode=True)
    )
    return config_path


def _run_diffsinger_command(ds_dir: Path, script: str, args: list[str]) -> None:
    """Run a DiffSinger script."""
    cmd = [sys.executable, str(ds_dir / "scripts" / script)] + args
    console.print(f"  [dim]$ {' '.join(cmd)}[/dim]")
    subprocess.run(cmd, cwd=str(ds_dir), check=True)


def run_train(project_dir: Path, config: ProjectConfig, target: str = "both") -> None:
    """Train DiffSinger models locally."""
    if config.training.backend == "colab":
        console.print("[yellow]Backend is set to 'colab'. Use 'dsmaker colab-export' instead.[/yellow]")
        console.print("  Or change backend to 'local' in dsmaker.yaml")
        return

    if not has_cuda():
        console.print("[yellow]Warning: No CUDA GPU detected. Training will be very slow on CPU.[/yellow]")
        console.print("  Consider using 'dsmaker colab-export' for Google Colab training.")

    ds_dir = ensure_diffsinger()
    console.print("[bold]Starting DiffSinger training...[/bold]")

    if target in ("acoustic", "both"):
        console.print("\n[bold cyan]== Acoustic Model ==[/bold cyan]")
        acoustic_config = _generate_acoustic_config(project_dir, config, ds_dir)

        console.print("  Binarizing data...")
        _run_diffsinger_command(ds_dir, "binarize.py", ["--config", str(acoustic_config)])

        console.print("  Training acoustic model...")
        _run_diffsinger_command(
            ds_dir,
            "train.py",
            [
                "--config", str(acoustic_config),
                "--exp_name", f"{config.name}_acoustic",
                "--reset",
            ],
        )

    if target in ("variance", "both"):
        console.print("\n[bold cyan]== Variance Model ==[/bold cyan]")
        variance_config = _generate_variance_config(project_dir, config, ds_dir)

        console.print("  Binarizing data...")
        _run_diffsinger_command(ds_dir, "binarize.py", ["--config", str(variance_config)])

        console.print("  Training variance model...")
        _run_diffsinger_command(
            ds_dir,
            "train.py",
            [
                "--config", str(variance_config),
                "--exp_name", f"{config.name}_variance",
                "--reset",
            ],
        )

    console.print("\n[green]Training complete![/green]")


def export_for_colab(project_dir: Path, config: ProjectConfig) -> Path:
    """Package dataset and configs into a zip for Google Colab training."""
    console.print("[bold]Preparing Colab export...[/bold]")

    ds_dir = ensure_diffsinger()

    # Generate configs
    acoustic_config = _generate_acoustic_config(project_dir, config, ds_dir)
    variance_config = _generate_variance_config(project_dir, config, ds_dir)

    # Create zip
    export_dir = ensure_dir(project_dir / "export")
    zip_path = export_dir / f"{config.name}_colab_data.zip"

    dataset_raw = project_dir / "dataset" / "raw"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add dataset files
        for wav in (dataset_raw / "wavs").glob("*.wav"):
            zf.write(wav, f"dataset/raw/wavs/{wav.name}")
        csv_path = dataset_raw / "transcriptions.csv"
        if csv_path.exists():
            zf.write(csv_path, "dataset/raw/transcriptions.csv")

        # Add config files
        zf.write(acoustic_config, "configs/acoustic_config.yaml")
        zf.write(variance_config, "configs/variance_config.yaml")

    console.print(f"[green]Colab data exported: {zip_path}[/green]")
    console.print(f"  Size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Generate Colab notebook
    notebook_path = _generate_colab_notebook(project_dir, config)
    console.print(f"  Notebook: {notebook_path}")

    return zip_path


def _generate_colab_notebook(project_dir: Path, config: ProjectConfig) -> Path:
    """Generate a Google Colab notebook for training."""
    export_dir = ensure_dir(project_dir / "export")
    notebook_path = export_dir / f"{config.name}_train.ipynb"

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# DiffSinger Training: {config.name}\n",
                "\n",
                "Auto-generated by dsmaker. Upload the zip file and run all cells.\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# 1. Setup\n",
                "!git clone --depth 1 https://github.com/openvpi/DiffSinger.git\n",
                "%cd DiffSinger\n",
                "!pip install -r requirements.txt -q\n",
            ],
            "outputs": [],
            "execution_count": None,
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# 2. Upload and extract data\n",
                "from google.colab import files\n",
                "import zipfile, os\n",
                "\n",
                f"# Upload {config.name}_colab_data.zip\n",
                "uploaded = files.upload()\n",
                "zip_name = list(uploaded.keys())[0]\n",
                "with zipfile.ZipFile(zip_name, 'r') as z:\n",
                "    z.extractall('.')\n",
                "print('Data extracted!')\n",
            ],
            "outputs": [],
            "execution_count": None,
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# 3. Train acoustic model\n",
                "!python scripts/binarize.py --config configs/acoustic_config.yaml\n",
                f"!python scripts/train.py --config configs/acoustic_config.yaml --exp_name {config.name}_acoustic --reset\n",
            ],
            "outputs": [],
            "execution_count": None,
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# 4. Train variance model\n",
                "!python scripts/binarize.py --config configs/variance_config.yaml\n",
                f"!python scripts/train.py --config configs/variance_config.yaml --exp_name {config.name}_variance --reset\n",
            ],
            "outputs": [],
            "execution_count": None,
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# 5. Export ONNX\n",
                f"!python scripts/export.py acoustic --exp {config.name}_acoustic\n",
                f"!python scripts/export.py variance --exp {config.name}_variance\n",
            ],
            "outputs": [],
            "execution_count": None,
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# 6. Download exported models\n",
                "import shutil\n",
                f"shutil.make_archive('{config.name}_onnx', 'zip', 'checkpoints')\n",
                f"files.download('{config.name}_onnx.zip')\n",
            ],
            "outputs": [],
            "execution_count": None,
        },
    ]

    import json
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "accelerator": "GPU",
        },
        "cells": cells,
    }

    notebook_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False))
    return notebook_path
