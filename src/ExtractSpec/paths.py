from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Optional

PKG = "ExtractSpec"                   # just for the env var name
ENVVAR = f"{PKG.upper()}_DATA"    # ISM2025_DATA

# --- Discovery helpers --------------------------------------------------------
def _is_project_root(p: Path) -> bool:
    return (p / "pyproject.toml").exists() or (p / ".git").exists()

def project_root(start: Optional[Path] = None) -> Path:
    """Walk upward from start (or CWD) until we find repo root."""
    here = (start or Path.cwd()).resolve()
    for p in (here, *here.parents):
        if _is_project_root(p):
            return p
    return here  # fallback: current dir if nothing found

def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

# --- Main public paths --------------------------------------------------------
def get_data_dir(start: Optional[Path] = None) -> Path:
    """Resolve a data directory with precedence:
       1) env ISM2025_DATA
       2) <repo>/user_data
       3) <repo>/data
       4) a sibling 'data' next to repo (../data)
       5) last resort: ~/data (auto-created)
    """
    # 1) Environment override
    env = os.getenv(ENVVAR)
    if env:
        p = Path(os.path.expandvars(os.path.expanduser(env))).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 2) Project-root-based guesses
    root = project_root(start)
    candidates = [
        root / "user_data",
        root / "data",
        root.parent / "data",   # e.g., repo and data are siblings
    ]
    chosen = first_existing(candidates)
    if chosen:
        return chosen.resolve()

    # 5) Final fallback
    fallback = Path.home() / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback.resolve()

def get_results_dir(start: Optional[Path] = None) -> Path:
    root = project_root(start)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    return results.resolve()

# Convenience aliases you can import elsewhere
DATA_DIR = get_data_dir()
RESULTS_DIR = get_results_dir()

# --- Useful extras ------------------------------------------------------------
from contextlib import contextmanager

@contextmanager
def cd(path: Path):
    """Temporarily change working directory."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)

def require(path: Path, msg: str = "") -> Path:
    """Assert a path exists and return it; raise with a helpful message otherwise."""
    if not Path(path).exists():
        hint = f"\nHint: set {ENVVAR}=/absolute/path/to/your/data" if ENVVAR not in os.environ else ""
        raise FileNotFoundError(msg or f"Missing: {path}{hint}")
    return Path(path)