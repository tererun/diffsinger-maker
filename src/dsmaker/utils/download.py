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

# Direct download URLs for SOFA pretrained models (GitHub Releases)
SOFA_MODEL_URLS = {
    "ja": {
        "repo": "Greenleaf2001/SOFA_Models",
        "tag": "JPN_Test2_Plus",
        "ckpt": "step.100000.ckpt",
        "dict": "japanese-extension-sofa.txt",
    },
    "en": {
        "repo": "spicytigermeat/SOFA_Models",
        "tag": "tgm_en_v100",
        "ckpt": "steps_200000.ckpt",
        "dict": None,
    },
    "zh": {
        "repo": "qiuqiao/SOFA",
        "tag": None,  # Use default opencpop dict in SOFA repo
        "ckpt": None,
        "dict": None,
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


def _gh_release_download_url(repo: str, tag: str, filename: str) -> str:
    """Build a direct download URL for a GitHub Release asset."""
    return f"https://github.com/{repo}/releases/download/{tag}/{filename}"


def download_file(url: str, target: Path) -> Path:
    """Download a file with progress bar."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"  Downloading {target.name}...")
    with Progress() as progress:
        task = progress.add_task("Downloading...", total=None)

        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                progress.update(task, total=total_size, completed=block_num * block_size)

        urllib.request.urlretrieve(url, str(target), reporthook=reporthook)
    return target


_SOFA_DEPS = [
    "lightning", "torch", "torchaudio", "scipy", "librosa",
    "matplotlib", "h5py", "tensorboard", "tensorboardX", "textgrid",
    "chardet", "numba", "einops", "pandas", "tqdm", "setuptools",
]


def ensure_sofa(cache_dir: Path | None = None) -> Path:
    cache = cache_dir or get_cache_dir()
    sofa_dir = cache / "SOFA"
    clone_repo(SOFA_REPO, sofa_dir)
    # Install SOFA dependencies (without strict version pins)
    marker = sofa_dir / ".deps_installed"
    if not marker.exists():
        console.print("  Installing SOFA dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + _SOFA_DEPS + ["-q"],
            check=True,
            capture_output=True,
            text=True,
        )
        marker.write_text("ok")
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
    """Get SOFA model path, downloading automatically if not present."""
    cache = cache_dir or get_cache_dir()
    models_dir = cache / "sofa_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing model
    for ckpt in models_dir.glob(f"*{language}*.ckpt"):
        return ckpt
    for ckpt in models_dir.glob("*.ckpt"):
        return ckpt

    # Auto-download
    model_info = SOFA_MODEL_URLS.get(language)
    if model_info and model_info.get("ckpt") and model_info.get("tag"):
        console.print(f"[bold]Downloading SOFA model for '{language}'...[/bold]")
        url = _gh_release_download_url(
            model_info["repo"], model_info["tag"], model_info["ckpt"]
        )
        dest = models_dir / f"{language}_{model_info['ckpt']}"
        download_file(url, dest)

        # Also download dictionary if available
        if model_info.get("dict"):
            dict_url = _gh_release_download_url(
                model_info["repo"], model_info["tag"], model_info["dict"]
            )
            dict_dest = models_dir / model_info["dict"]
            download_file(dict_url, dict_dest)

        console.print(f"[green]SOFA model downloaded: {dest.name}[/green]")
        return dest

    # No auto-download available for this language
    info = model_info.get("info", "") if model_info else ""
    if not info:
        info = f"Download from https://github.com/qiuqiao/SOFA/discussions — search for {language} model"
    console.print(f"[yellow]SOFA model for '{language}' not found.[/yellow]")
    console.print(f"  {info}")
    console.print(f"  Place the .ckpt file in: [bold]{models_dir}[/bold]")
    raise FileNotFoundError(
        f"SOFA model not found for language '{language}'. "
        f"Place a .ckpt file in {models_dir}"
    )
