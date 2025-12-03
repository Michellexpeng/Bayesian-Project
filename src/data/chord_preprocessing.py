"""Chord preprocessing utilities for POP909.

- Normalize chord symbols (transposition, quality simplification)
- Build vocabulary
- Align chords to beat grid
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import re


# Simple chord root to semitone mapping (relative to C)
ROOT_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}

SEMITONE_TO_ROOT = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def parse_chord_label(label: str) -> Tuple[str | None, str]:
    """Parse 'Root:quality' -> (root, quality).
    
    Examples:
      'F#:maj' -> ('F#', 'maj')
      'Bb:min' -> ('Bb', 'min')
      'N' -> (None, 'N')
    """
    label = label.strip()
    if label == "N" or not label:
        return (None, "N")
    if ":" in label:
        root, quality = label.split(":", 1)
        return (root.strip(), quality.strip())
    # Fallback: treat entire label as root + implicit major
    return (label, "maj")


def transpose_chord(label: str, semitones: int) -> str:
    """Transpose a chord by semitones.
    
    E.g., transpose_chord('F#:maj', -6) -> 'C:maj'
    """
    root, quality = parse_chord_label(label)
    if root is None:
        return "N"
    old_semi = ROOT_TO_SEMITONE.get(root)
    if old_semi is None:
        return label  # unknown root, keep as-is
    new_semi = (old_semi + semitones) % 12
    new_root = SEMITONE_TO_ROOT[new_semi]
    return f"{new_root}:{quality}"


def simplify_quality(quality: str) -> str:
    """Simplify chord quality to a small set: maj, min, dom7, maj7, min7, dim, aug, sus4, N."""
    quality = quality.lower()
    # Map common patterns
    if quality in {"maj", "major", ""}:
        return "maj"
    if quality in {"min", "minor"}:
        return "min"
    if quality in {"7", "dom7"}:
        return "dom7"
    if "maj7" in quality:
        return "maj7"
    if "min7" in quality:
        return "min7"
    if "dim" in quality:
        return "dim"
    if "aug" in quality:
        return "aug"
    if "sus4" in quality:
        return "sus4"
    if quality == "n":
        return "N"
    # Default: keep as-is but mark unknown
    return quality


def normalize_chord(label: str, transpose_semitones: int = 0) -> str:
    """Transpose + simplify quality."""
    transposed = transpose_chord(label, transpose_semitones)
    root, quality = parse_chord_label(transposed)
    if root is None:
        return "N"
    quality_simp = simplify_quality(quality)
    return f"{root}:{quality_simp}"


def build_vocabulary(chord_labels: List[str]) -> Dict[str, int]:
    """Build a chord -> int mapping from a list of chord labels."""
    unique = sorted(set(chord_labels))
    return {ch: i for i, ch in enumerate(unique)}


def align_chords_to_beats(
    chords: List[Tuple[float, float, str]], beat_times: List[float]
) -> List[str]:
    """Assign one chord label per beat by majority overlap or nearest."""
    result = []
    for bt in beat_times:
        # Find chord that overlaps this beat time
        best = "N"
        for start, end, label in chords:
            if start <= bt < end:
                best = label
                break
        result.append(best)
    return result
