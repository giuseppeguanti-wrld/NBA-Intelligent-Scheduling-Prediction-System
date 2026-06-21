import tomllib
from pathlib import Path
from typing import Any

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.toml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "rb") as f:
        return tomllib.load(f)
