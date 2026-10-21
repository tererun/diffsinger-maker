"""Project configuration management."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


CONFIG_FILENAME = "dsmaker.yaml"


@dataclass
class SliceConfig:
    threshold: float = -40.0
    min_length: int = 5000
    min_interval: int = 300
    hop_size: int = 10
    max_sil_kept: int = 500


@dataclass
class SofaConfig:
    model: str = "auto"


@dataclass
class TrainingTargetConfig:
    max_updates: int = 100000
    batch_size: int = 8


@dataclass
class TrainingConfig:
    backend: str = "local"  # "local" or "colab"
    acoustic: TrainingTargetConfig = field(default_factory=TrainingTargetConfig)
    variance: TrainingTargetConfig = field(
        default_factory=lambda: TrainingTargetConfig(max_updates=80000)
    )


@dataclass
class ExportConfig:
    vocoder: str = "nsf_hifigan"


@dataclass
class PackageConfig:
    author: str = ""
    version: str = "1.0.0"
    image: str = ""


@dataclass
class ProjectConfig:
    name: str = "MySinger"
    language: str = "ja"
    sample_rate: int = 44100
    audio_dir: str = "raw/songs"
    lyrics_dir: str = "raw/lyrics"
    slice: SliceConfig = field(default_factory=SliceConfig)
    sofa: SofaConfig = field(default_factory=SofaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    package: PackageConfig = field(default_factory=PackageConfig)

    def save(self, project_dir: Path) -> Path:
        path = project_dir / CONFIG_FILENAME
        data = asdict(self)
        path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
        return path

    @classmethod
    def load(cls, project_dir: Path) -> ProjectConfig:
        path = project_dir / CONFIG_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        data = yaml.safe_load(path.read_text())
        return _dict_to_config(data)


def _dict_to_config(d: dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(
        name=d.get("name", "MySinger"),
        language=d.get("language", "ja"),
        sample_rate=d.get("sample_rate", 44100),
        audio_dir=d.get("audio_dir", "raw/songs"),
        lyrics_dir=d.get("lyrics_dir", "raw/lyrics"),
        slice=SliceConfig(**d.get("slice", {})),
        sofa=SofaConfig(**d.get("sofa", {})),
        training=TrainingConfig(
            backend=d.get("training", {}).get("backend", "local"),
            acoustic=TrainingTargetConfig(
                **d.get("training", {}).get("acoustic", {})
            ),
            variance=TrainingTargetConfig(
                **d.get("training", {}).get("variance", {"max_updates": 80000})
            ),
        ),
        export=ExportConfig(**d.get("export", {})),
        package=PackageConfig(**d.get("package", {})),
    )
