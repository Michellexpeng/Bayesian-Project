"""Utility helpers for loading and inspecting MIDI files.

Imports are optional to keep the skeleton runnable without external deps.
Install extras (`mido`) via `pip install -r requirements.txt`.
"""
from pathlib import Path
from typing import List

try:
    from mido import MidiFile  # type: ignore
except Exception:  # pragma: no cover - soft dependency
    MidiFile = None  # type: ignore


def list_midi_files(root: Path, max_files: int | None = None) -> List[Path]:
    files = sorted(root.glob("**/*.mid"))
    if max_files is not None:
        files = files[:max_files]
    return files


def basic_summary(path: Path) -> dict:
    if MidiFile is None:
        raise RuntimeError(
            "mido is not installed. Please `pip install -r requirements.txt` to enable MIDI parsing."
        )
    mf = MidiFile(path)
    return {
        "tracks": len(mf.tracks),
        "length_seconds": mf.length,
        "file": str(path)
    }

if __name__ == "__main__":
    # no-op; this module provides helpers only
    pass
