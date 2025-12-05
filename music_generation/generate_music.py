"""Generate chord progression and export to MIDI.

使用训练好的HMM模型生成新的和弦序列，并转换为MIDI文件。

Usage:
  python scripts/generate_music.py --model models/hmm_conditional.pkl --mode major --length 32 --output output.mid
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pickle
from midiutil import MIDIFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sample_chord(prob_dist, vocab_inv):
    """Sample a chord from probability distribution."""
    chord_idx = np.random.choice(len(prob_dist), p=prob_dist)
    return vocab_inv[chord_idx]


def generate_chord_sequence(params, mode='major', length=32, seed=None):
    """Generate a chord sequence using the HMM model.
    
    Args:
        params: Model parameters (dict with start_prob, trans_prob, vocab, etc.)
        mode: 'major' or 'minor'
        length: Number of chords to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of chord labels (functional chords like 'I', 'V', 'vi', etc.)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get mode-specific parameters
    if mode == 'major':
        start_prob = params['major_start_prob']
        trans_prob = params['major_trans_prob']
    elif mode == 'minor':
        start_prob = params['minor_start_prob']
        trans_prob = params['minor_trans_prob']
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    vocab = params['vocab']
    vocab_inv = params.get('vocab_inv', params.get('inv_vocab', {}))
    
    # Sample initial chord
    current_chord = sample_chord(start_prob, vocab_inv)
    sequence = [current_chord]
    
    # Generate remaining chords
    for _ in range(length - 1):
        current_idx = vocab[current_chord]
        next_chord = sample_chord(trans_prob[current_idx], vocab_inv)
        sequence.append(next_chord)
        current_chord = next_chord
    
    return sequence


def roman_to_midi_notes(roman_numeral, key_root=60, mode='major'):
    """Convert Roman numeral to MIDI notes (triad).
    
    Args:
        roman_numeral: e.g., 'I', 'ii', 'V7', 'vi', 'bVII'
        key_root: MIDI note number for key root (60 = C4)
        mode: 'major' or 'minor'
        
    Returns:
        List of MIDI note numbers for the chord
    """
    # Major scale intervals from root
    major_scale = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B
    # Natural minor scale intervals
    minor_scale = [0, 2, 3, 5, 7, 8, 10]  # C D Eb F G Ab Bb
    
    scale = major_scale if mode == 'major' else minor_scale
    
    # Parse Roman numeral
    numeral = roman_numeral.upper().replace('M', '').replace('7', '')
    
    # Handle chromatic alterations
    alteration = 0
    if numeral.startswith('B'):
        alteration = -1
        numeral = numeral[1:]
    elif numeral.startswith('#'):
        alteration = 1
        numeral = numeral[1:]
    
    # Map Roman to scale degree (0-indexed)
    roman_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}
    
    if numeral not in roman_map:
        # Handle unknown chords - default to I
        degree = 0
    else:
        degree = roman_map[numeral]
    
    # Determine chord quality (major or minor)
    is_minor = roman_numeral[0].islower() if roman_numeral else False
    
    # Build triad
    root = key_root + scale[degree] + alteration
    third = root + (3 if is_minor else 4)  # minor 3rd or major 3rd
    fifth = root + 7  # perfect 5th
    
    return [root, third, fifth]


def create_midi_from_chords(chord_sequence, output_path, mode='major', key_root=60, tempo=120):
    """Create MIDI file from chord sequence.
    
    Args:
        chord_sequence: List of functional chord labels
        output_path: Output MIDI file path
        mode: 'major' or 'minor'
        key_root: MIDI note for key root (60=C, 62=D, 64=E, etc.)
        tempo: BPM
    """
    # Create MIDI file with 1 track
    midi = MIDIFile(1)
    track = 0
    channel = 0
    time = 0
    duration = 2  # 2 beats per chord
    volume = 100
    
    midi.addTempo(track, time, tempo)
    
    # Add chords to MIDI
    for chord_label in chord_sequence:
        notes = roman_to_midi_notes(chord_label, key_root, mode)
        
        # Add each note in the chord
        for note in notes:
            midi.addNote(track, channel, note, time, duration, volume)
        
        time += duration
    
    # Write MIDI file
    with open(output_path, 'wb') as f:
        midi.writeFile(f)
    
    print(f"✅ MIDI file created: {output_path}")
    print(f"   - {len(chord_sequence)} chords")
    print(f"   - Duration: {len(chord_sequence) * duration} beats")
    print(f"   - Mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Generate chord progression and export to MIDI")
    parser.add_argument('--model', type=str, required=True, help='Path to trained HMM model (.pkl)')
    parser.add_argument('--mode', type=str, default='major', choices=['major', 'minor'], 
                        help='Key mode (major or minor)')
    parser.add_argument('--length', type=int, default=32, help='Number of chords to generate')
    parser.add_argument('--output', type=str, default='generated_music.mid', help='Output MIDI file path')
    parser.add_argument('--key', type=int, default=60, help='Key root MIDI note (60=C, 62=D, 64=E, etc.)')
    parser.add_argument('--tempo', type=int, default=120, help='Tempo in BPM')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load model
    print(f"📂 Loading model from {args.model}...")
    with open(args.model, 'rb') as f:
        params = pickle.load(f)
    
    print(f"✅ Model loaded")
    print(f"   - Vocabulary size: {len(params['vocab'])}")
    
    # Generate chord sequence
    print(f"\n🎵 Generating {args.length} chords in {args.mode} mode...")
    chord_sequence = generate_chord_sequence(params, mode=args.mode, length=args.length, seed=args.seed)
    
    print(f"✅ Chord sequence generated:")
    print(f"   {' - '.join(chord_sequence)}")
    
    # Create MIDI file
    print(f"\n🎹 Creating MIDI file...")
    create_midi_from_chords(chord_sequence, args.output, mode=args.mode, key_root=args.key, tempo=args.tempo)
    
    print(f"\n✨ Done! You can now:")
    print(f"   1. Open {args.output} in GarageBand")
    print(f"   2. Open {args.output} in any MIDI player")
    print(f"   3. Import {args.output} into your DAW")


if __name__ == '__main__':
    main()
