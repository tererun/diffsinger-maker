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


def _generate_colab_config(config: ProjectConfig, model_type: str) -> dict:
    """Generate a DiffSinger config for Colab (relative paths, no local deps)."""
    # Paths relative to DiffSinger repo root on Colab
    base_yaml = f"configs/{model_type}.yaml"
    raw_dir = "dataset/raw"
    binary_dir = f"dataset/binary_{model_type}" if model_type == "variance" else "dataset/binary"

    cfg: dict = {
        "base_config": [base_yaml],
        "datasets": [{
            "raw_data_dir": raw_dir,
            "speaker": config.name,
            "spk_id": 0,
            "language": config.language,
            "test_prefixes": [],
        }],
        "binary_data_dir": binary_dir,
        "dictionary": f"dictionaries/{config.language}.txt",
        "dictionaries": {config.language: f"dictionaries/{config.language}.txt"},
        "num_spk": 1,
        "audio_sample_rate": config.sample_rate,
    }

    if model_type == "acoustic":
        cfg.update({
            "audio_num_mel_bins": 128,
            "hop_size": 512, "fft_size": 2048, "win_size": 2048,
            "max_batch_size": config.training.acoustic.batch_size,
            "max_updates": config.training.acoustic.max_updates,
            "accumulate_grad_batches": 1,
            "val_check_interval": 2000,
            "num_valid_plots": 4,
            "backbone_type": "lynxnet",
            "diffusion_type": "reflow",
            "use_spk_id": False,
        })
    else:
        cfg.update({
            "max_batch_size": config.training.variance.batch_size,
            "max_updates": config.training.variance.max_updates,
            "predict_dur": True,
            "predict_pitch": True,
            "predict_voicing": True,
            "predict_tension": True,
        })

    return cfg


def export_for_colab(project_dir: Path, config: ProjectConfig) -> Path:
    """Package dataset and configs into a zip for Google Colab training.

    Does NOT require DiffSinger to be cloned locally.
    """
    console.print("[bold]Preparing Colab export...[/bold]")

    export_dir = ensure_dir(project_dir / "export")
    zip_path = export_dir / f"{config.name}_colab_data.zip"
    dataset_raw = project_dir / "dataset" / "raw"

    # Generate Colab-oriented configs (relative paths for Colab)
    acoustic_cfg = _generate_colab_config(config, "acoustic")
    variance_cfg = _generate_colab_config(config, "variance")

    # Pick test prefixes
    csv_path = dataset_raw / "transcriptions.csv"
    if csv_path.exists():
        import csv as csv_mod
        with open(csv_path, encoding="utf-8") as f:
            names = [row["name"] for row in csv_mod.DictReader(f)]
        if names:
            acoustic_cfg["datasets"][0]["test_prefixes"] = names[:3]
            variance_cfg["datasets"][0]["test_prefixes"] = names[:3]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Dataset files
        for wav in (dataset_raw / "wavs").glob("*.wav"):
            zf.write(wav, f"dataset/raw/wavs/{wav.name}")
        if csv_path.exists():
            zf.write(csv_path, "dataset/raw/transcriptions.csv")

        # Config files
        zf.writestr(
            "configs/acoustic_config.yaml",
            yaml.dump(acoustic_cfg, default_flow_style=False, allow_unicode=True),
        )
        zf.writestr(
            "configs/variance_config.yaml",
            yaml.dump(variance_cfg, default_flow_style=False, allow_unicode=True),
        )

        # Dictionary file — generate from transcriptions.csv phonemes
        if csv_path.exists():
            import csv as csv_mod
            phonemes = set()
            with open(csv_path, encoding="utf-8") as f:
                for row in csv_mod.DictReader(f):
                    phonemes.update(row["ph_seq"].split())
            dict_content = "\n".join(f"{ph}\t{ph}" for ph in sorted(phonemes))
            zf.writestr(f"dictionaries/{config.language}.txt", dict_content)

    console.print(f"[green]Colab data exported: {zip_path}[/green]")
    console.print(f"  Size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Generate notebook
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
                "# Install deps without strict version pins (requirements.txt has conflicts)\n",
                "!pip install 'lightning<2.4' torch torchaudio tensorboard scipy matplotlib h5py onnx onnxsim praat-parselmouth resampy torchcrepe pyloudnorm pyworld pyyaml tqdm -q\n",
                "\n",
                "# Download pretrained models\n",
                "!mkdir -p checkpoints/rmvpe checkpoints/vr\n",
                "!wget -q https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip && unzip -q rmvpe.zip -d checkpoints/rmvpe && rm rmvpe.zip\n",
                "!wget -q https://github.com/yxlllc/vocal-remover/releases/download/hnsep_240512/hnsep_240512.zip && unzip -q hnsep_240512.zip -d checkpoints/vr && rm hnsep_240512.zip\n",
                "!wget -q https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip && unzip -q pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip -d checkpoints && rm pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip\n",
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
