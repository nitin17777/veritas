from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.is_absolute():
        path = ROOT_DIR / path

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config