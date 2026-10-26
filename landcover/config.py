"""读取项目配置。路径一律相对项目根目录，不写死本机盘符。"""

from __future__ import annotations

from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    root = project_root()
    config_path = Path(path) if path else root / "config" / "config.yaml"
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_root"] = str(root)
    config["_config_path"] = str(config_path)
    return config


def resolve(config: dict, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return Path(config["_root"]) / path
