"""POP909 dataset utilities (skeleton).

Assumes the POP909 root contains per-song folders with MIDI files.
This module provides simple discovery and placeholders for future parsing.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

POP909_DEFAULT_GLOBS = ["**/*.mid", "**/*.midi"]


def find_songs(root: Path, patterns: List[str] | None = None) -> List[Path]:
    root = Path(root)
    patterns = patterns or POP909_DEFAULT_GLOBS
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.glob(pat))
    # Deduplicate and sort
    files = sorted({p.resolve() for p in files})
    return files


def describe(root: Path, limit: int | None = 5) -> List[str]:
    files = find_songs(root)
    if limit is not None:
        files = files[:limit]
    return [str(p) for p in files]
