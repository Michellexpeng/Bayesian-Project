"""POP909 parser: load chord, beat, key annotations per song.

Each song folder contains:
- chord_midi.txt: (start_time, end_time, chord_label)
- beat_midi.txt: (time, is_downbeat, is_beat)
- key_audio.txt: (start, end, key_label)
- *.mid: full MIDI file

This module provides loaders for annotations (not yet parsing MIDI tracks).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import re


@dataclass
class ChordAnnotation:
    start: float
    end: float
    label: str  # e.g., "F#:maj", "Bb:min", "N"


@dataclass
class BeatAnnotation:
    time: float
    is_downbeat: bool
    is_beat: bool


@dataclass
class KeyAnnotation:
    start: float
    end: float
    label: str  # e.g., "Gb:maj"


@dataclass
class SongAnnotations:
    song_id: str
    chords: List[ChordAnnotation]
    beats: List[BeatAnnotation]
    key: KeyAnnotation | None
    midi_path: Path | None


def load_chords(path: Path) -> List[ChordAnnotation]:
    """Parse chord_midi.txt: three tab-separated columns."""
    chords = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
            chords.append(ChordAnnotation(start, end, label))
    return chords


def load_beats(path: Path) -> List[BeatAnnotation]:
    """Parse beat_midi.txt: space-separated (time, downbeat, beat)."""
    beats = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            time = float(parts[0])
            is_downbeat = float(parts[1]) > 0.5
            is_beat = float(parts[2]) > 0.5
            beats.append(BeatAnnotation(time, is_downbeat, is_beat))
    return beats


def load_key(path: Path) -> KeyAnnotation | None:
    """Parse key_audio.txt: typically one line (start, end, key_label)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
            return KeyAnnotation(start, end, label)
    return None


def load_song(song_dir: Path) -> SongAnnotations:
    """Load all annotations for one song folder."""
    song_id = song_dir.name
    chord_path = song_dir / "chord_midi.txt"
    beat_path = song_dir / "beat_midi.txt"
    key_path = song_dir / "key_audio.txt"
    midi_candidates = list(song_dir.glob("*.mid"))
    midi_path = midi_candidates[0] if midi_candidates else None

    chords = load_chords(chord_path) if chord_path.exists() else []
    beats = load_beats(beat_path) if beat_path.exists() else []
    key = load_key(key_path) if key_path.exists() else None

    return SongAnnotations(
        song_id=song_id, chords=chords, beats=beats, key=key, midi_path=midi_path
    )


def find_songs(root: Path) -> List[Path]:
    """Find all song subdirectories (numeric folder names)."""
    root = Path(root)
    candidates = [p for p in root.iterdir() if p.is_dir() and re.match(r"^\d+$", p.name)]
    return sorted(candidates)


def load_dataset(root: Path, limit: int | None = None) -> List[SongAnnotations]:
    """Load annotations for all songs in the POP909 root."""
    song_dirs = find_songs(root)
    if limit is not None:
        song_dirs = song_dirs[:limit]
    return [load_song(d) for d in song_dirs]
