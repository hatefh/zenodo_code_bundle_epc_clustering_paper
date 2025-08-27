# code/utils/paths.py
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

def pth(*parts: str) -> str:
    """Join one or more path components relative to the repository root."""
    return str(_REPO_ROOT.joinpath(*parts))
