"""Minimal .env loader (no python-dotenv dependency)."""

import os
from pathlib import Path


def load_env(path: Path = Path(".env")) -> None:
    """Export KEY=VALUE lines from `path` for variables not already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip().strip("\"'")
            if value:
                os.environ.setdefault(key.strip(), value)
