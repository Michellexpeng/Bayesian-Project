"""Key-aware feature extraction for chord sequences.

This module adds harmonic function and scale degree information
on top of basic chord labels, enabling key-aware modeling.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path

# Add project root to path for imports
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chord_preprocessing import parse_chord_label, ROOT_TO_SEMITONE, SEMITONE_TO_ROOT


# Roman numeral mapping for scale degrees
MAJOR_SCALE_DEGREES = {
    0: "I",    # Tonic
    2: "II",   # Supertonic
    4: "III",  # Mediant
    5: "IV",   # Subdominant
    7: "V",    # Dominant
    9: "VI",   # Submediant
    11: "VII"  # Leading tone
}

MINOR_SCALE_DEGREES = {
    0: "i",    # Tonic
    2: "ii",   # Supertonic
    3: "III",  # Mediant (relative major)
    5: "iv",   # Subdominant
    7: "v",    # Dominant (or V if harmonic minor)
    8: "VI",   # Submediant (relative major)
    10: "VII"  # Subtonic
}


def get_scale_degree(chord_root: str, key_root: str, is_minor: bool = False) -> Tuple[int, str]:
    """Calculate scale degree of a chord relative to key.
    
    Args:
        chord_root: Root note of the chord (e.g., "F")
        key_root: Root note of the key (e.g., "C")
        is_minor: Whether the key is minor
        
    Returns:
        (semitone_distance, roman_numeral)
        
    Examples:
        >>> get_scale_degree("F", "C", False)
        (5, "IV")
        >>> get_scale_degree("D", "C", False)
        (2, "II")
        >>> get_scale_degree("E", "A", True)
        (7, "v")
    """
    if chord_root not in ROOT_TO_SEMITONE or key_root not in ROOT_TO_SEMITONE:
        return (0, "?")
    
    # Calculate semitone distance from key tonic
    chord_semitone = ROOT_TO_SEMITONE[chord_root]
    key_semitone = ROOT_TO_SEMITONE[key_root]
    distance = (chord_semitone - key_semitone) % 12
    
    # Get roman numeral
    degrees = MINOR_SCALE_DEGREES if is_minor else MAJOR_SCALE_DEGREES
    roman = degrees.get(distance, f"?{distance}")
    
    return (distance, roman)


def get_harmonic_function(scale_degree: int, is_minor: bool = False) -> str:
    """Classify chord's harmonic function based on scale degree.
    
    Args:
        scale_degree: Semitone distance from tonic (0-11)
        is_minor: Whether in minor key
        
    Returns:
        Function category: "tonic", "subdominant", "dominant", "mediant", or "other"
    """
    if is_minor:
        # Minor key functions
        if scale_degree in [0, 3, 8]:  # i, III, VI
            return "tonic"
        elif scale_degree in [5, 10]:  # iv, VII
            return "subdominant"
        elif scale_degree in [7, 2]:   # v/V, ii°
            return "dominant"
        elif scale_degree == 3:        # III
            return "mediant"
    else:
        # Major key functions
        if scale_degree in [0, 4, 9]:  # I, iii, vi
            return "tonic"
        elif scale_degree in [5, 2]:   # IV, ii
            return "subdominant"
        elif scale_degree in [7, 11]:  # V, vii°
            return "dominant"
        elif scale_degree == 4:        # iii
            return "mediant"
    
    return "other"


def extract_key_aware_features(
    chord_label: str, 
    key_label: str
) -> Dict[str, any]:
    """Extract rich harmonic features for a chord in context of a key.
    
    Args:
        chord_label: Chord label (e.g., "F:maj", "D:min")
        key_label: Key label (e.g., "C:maj", "A:min")
        
    Returns:
        Dictionary with:
        - chord_root: Root note
        - chord_quality: Quality (maj, min, dom7, etc.)
        - key_root: Key tonic
        - key_mode: "major" or "minor"
        - scale_degree: Semitone distance from tonic (0-11)
        - roman_numeral: Roman numeral representation
        - function: Harmonic function category
        
    Example:
        >>> extract_key_aware_features("F:maj", "C:maj")
        {
            'chord_root': 'F',
            'chord_quality': 'maj',
            'key_root': 'C',
            'key_mode': 'major',
            'scale_degree': 5,
            'roman_numeral': 'IV',
            'function': 'subdominant'
        }
    """
    # Parse chord
    chord_root, chord_quality = parse_chord_label(chord_label)
    
    # Parse key
    key_root, key_mode_str = parse_chord_label(key_label)
    is_minor = "min" in key_mode_str.lower() if key_mode_str else False
    
    # Handle N (no chord) or unknown
    if chord_root is None or key_root is None:
        return {
            'chord_root': None,
            'chord_quality': chord_quality,
            'key_root': key_root,
            'key_mode': 'minor' if is_minor else 'major',
            'scale_degree': None,
            'roman_numeral': 'N',
            'function': 'none'
        }
    
    # Calculate scale degree and function
    scale_degree, roman = get_scale_degree(chord_root, key_root, is_minor)
    function = get_harmonic_function(scale_degree, is_minor)
    
    return {
        'chord_root': chord_root,
        'chord_quality': chord_quality,
        'key_root': key_root,
        'key_mode': 'minor' if is_minor else 'major',
        'scale_degree': scale_degree,
        'roman_numeral': roman,
        'function': function
    }


def build_functional_vocabulary(chord_features_list: List[Dict]) -> Dict[str, int]:
    """Build vocabulary from functional representations.
    
    Instead of just chord labels, encode as "roman_numeral:quality"
    to capture harmonic function.
    
    Args:
        chord_features_list: List of feature dicts from extract_key_aware_features
        
    Returns:
        Vocabulary mapping "function_label" -> int
        
    Example:
        Input: [{'roman_numeral': 'IV', 'chord_quality': 'maj'}, ...]
        Output: {'IV:maj': 0, 'V:maj': 1, 'i:min': 2, ...}
    """
    functional_labels = []
    
    for features in chord_features_list:
        roman = features['roman_numeral']
        quality = features['chord_quality']
        
        if roman and roman != 'N':
            label = f"{roman}:{quality}"
        else:
            label = "N"
        
        functional_labels.append(label)
    
    # Build vocab
    unique_labels = sorted(set(functional_labels))
    return {label: idx for idx, label in enumerate(unique_labels)}


def chord_sequence_to_features(
    chords: List[str],
    key: str
) -> List[Dict]:
    """Convert chord sequence to key-aware feature dicts.
    
    Args:
        chords: List of chord labels
        key: Key label
        
    Returns:
        List of feature dictionaries
    """
    return [extract_key_aware_features(chord, key) for chord in chords]


if __name__ == "__main__":
    # Test examples
    print("Testing key-aware feature extraction...")
    
    # Example 1: C major progression
    print("\nExample 1: I-IV-V-I in C major")
    progression = ["C:maj", "F:maj", "G:maj", "C:maj"]
    key = "C:maj"
    
    for chord in progression:
        features = extract_key_aware_features(chord, key)
        print(f"  {chord:10s} → {features['roman_numeral']:4s} ({features['function']})")
    
    # Example 2: Same chords in different key
    print("\nExample 2: Same chords in A minor")
    key = "A:min"
    
    for chord in progression:
        features = extract_key_aware_features(chord, key)
        print(f"  {chord:10s} → {features['roman_numeral']:4s} ({features['function']})")
    
    # Example 3: Minor key progression
    print("\nExample 3: i-iv-V-i in D minor")
    progression = ["D:min", "G:min", "A:maj", "D:min"]
    key = "D:min"
    
    for chord in progression:
        features = extract_key_aware_features(chord, key)
        print(f"  {chord:10s} → {features['roman_numeral']:4s} ({features['function']})")
