"""Model and dependency download utilities."""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

console = Console()

SOFA_REPO = "https://github.com/qiuqiao/SOFA.git"
DIFFSINGER_REPO = "https://github.com/openvpi/DiffSinger.git"

# SOFA pretrained model URLs (community models)
SOFA_MODELS = {
    "ja": {
        "name": "Japanese SOFA Model",
        "info": "Download from https://github.com/qiuqiao/SOFA/discussions — search for Japanese model",
    },
    "en": {
        "name": "English SOFA Model (tgm_en)",
        "info": "Download from https://github.com/qiuqiao/SOFA/discussions — search for tgm_en",
    },
    "zh": {
        "name": "Mandarin SOFA Model",
        "info": "Download from https://github.com/qiuqiao/SOFA/discussions — search for Mandarin model",
    },
}


def get_cache_dir() -> Path:
    cache = Path.home() / ".cache" / "dsmaker"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def clone_repo(url: str, target: Path, branch: str | None = None) -> Path:
    if target.exists():
        console.print(f"  [dim]Already cloned: {target.name}[/dim]")
        return target
    console.print(f"  Cloning {url}...")
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(target)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return target


def ensure_sofa(cache_dir: Path | None = None) -> Path:
    cache = cache_dir or get_cache_dir()
    sofa_dir = cache / "SOFA"
    clone_repo(SOFA_REPO, sofa_dir)
    # Install SOFA dependencies
    req_file = sofa_dir / "requirements.txt"
    if req_file.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            check=True,
            capture_output=True,
            text=True,
        )
    return sofa_dir


def ensure_diffsinger(cache_dir: Path | None = None) -> Path:
    cache = cache_dir or get_cache_dir()
    ds_dir = cache / "DiffSinger"
    clone_repo(DIFFSINGER_REPO, ds_dir)
    req_file = ds_dir / "requirements.txt"
    if req_file.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            check=True,
            capture_output=True,
            text=True,
        )
    return ds_dir


def get_sofa_model_path(language: str, cache_dir: Path | None = None) -> Path:
    cache = cache_dir or get_cache_dir()
    models_dir = cache / "sofa_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Look for existing model files
    for ckpt in models_dir.glob(f"*{language}*.ckpt"):
        return ckpt
    for ckpt in models_dir.glob("*.ckpt"):
        return ckpt

    # No model found — guide user
    info = SOFA_MODELS.get(language, SOFA_MODELS.get("ja", {}))
    console.print(f"[yellow]SOFA model for '{language}' not found.[/yellow]")
    console.print(f"  {info.get('info', 'Please download a SOFA model.')}")
    console.print(f"  Place the .ckpt file in: [bold]{models_dir}[/bold]")
    raise FileNotFoundError(
        f"SOFA model not found for language '{language}'. "
        f"Place a .ckpt file in {models_dir}"
    )


def download_file(url: str, target: Path) -> Path:
    if target.exists():
        return target
    console.print(f"  Downloading {target.name}...")
    with Progress() as progress:
        task = progress.add_task("Downloading...", total=None)

        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                progress.update(task, total=total_size, completed=block_num * block_size)

        urllib.request.urlretrieve(url, str(target), reporthook=reporthook)
    return target
