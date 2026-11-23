"""
config.py
Loads config.yaml once and gives the rest of the project easy access to it.
"""
from functools import lru_cache
from pathlib import Path

import yaml

# Folder that contains config.yaml (the project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Sections that must exist, so we fail early with a clear message
REQUIRED_SECTIONS = [
    "project", "paths", "ingestion", "embedding", "retrieval",
    "llm", "rag", "cache", "logging",
]


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read config.yaml and return it as a dictionary (cached after first read)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    missing = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(f"config.yaml is missing sections: {missing}")
    return config


def resolve_path(relative_path: str) -> Path:
    """Turn a path from config.yaml (like 'data/pdfs') into a full path."""
    return PROJECT_ROOT / relative_path