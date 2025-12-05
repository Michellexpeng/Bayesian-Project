"""Advanced music generator with melody and rhythm.

生成更完整的音乐，包括：
- 和弦进行 (从HMM模型生成)
- 简单旋律 (基于和弦音)
- 节奏变化

Usage:
  python scripts/generate_full_music.py --model models/hmm_conditional.pkl --mode major --bars 8 --output my_song.mid
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
    """Generate a chord sequence using the HMM model."""
    if seed is not None:
        np.random.seed(seed)
    
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
    
    current_chord = sample_chord(start_prob, vocab_inv)
    sequence = [current_chord]
    
    for _ in range(length - 1):
        current_idx = vocab[current_chord]
        next_chord = sample_chord(trans_prob[current_idx], vocab_inv)
        sequence.append(next_chord)
        current_chord = next_chord
    
    return sequence


def roman_to_scale_degree(roman_numeral):
    """Parse Roman numeral to scale degree and quality.
    
    Returns:
        (degree, alteration, is_minor) tuple
    """
    numeral = roman_numeral.upper().replace('M', '').replace('7', '')
    
    alteration = 0
    if numeral.startswith('B'):
        alteration = -1
        numeral = numeral[1:]
    elif numeral.startswith('#'):
        alteration = 1
        numeral = numeral[1:]
    
    roman_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}
    degree = roman_map.get(numeral, 0)
    
    is_minor = roman_numeral[0].islower() if roman_numeral else False
    
    return degree, alteration, is_minor


def get_chord_notes(roman_numeral, key_root=60, mode='major', octave_offset=0):
    """Get MIDI notes for a chord.
    
    Args:
        roman_numeral: Functional chord label
        key_root: MIDI note for key root
        mode: 'major' or 'minor'
        octave_offset: Octave adjustment (-1, 0, 1, etc.)
        
    Returns:
        List of MIDI note numbers
    """
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    minor_scale = [0, 2, 3, 5, 7, 8, 10]
    
    scale = major_scale if mode == 'major' else minor_scale
    
    degree, alteration, is_minor = roman_to_scale_degree(roman_numeral)
    
    root = key_root + scale[degree] + alteration + (12 * octave_offset)
    third = root + (3 if is_minor else 4)
    fifth = root + 7
    
    return [root, third, fifth]


def get_chord_scale_notes(roman_numeral, key_root=60, mode='major', octave_offset=1):
    """Get scale notes that work with this chord (for melody).
    
    Returns a list of MIDI notes in the scale that are consonant with the chord.
    """
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    minor_scale = [0, 2, 3, 5, 7, 8, 10]
    
    scale = major_scale if mode == 'major' else minor_scale
    
    # Get all scale notes in the octave above
    base = key_root + (12 * octave_offset)
    scale_notes = [base + interval for interval in scale]
    
    return scale_notes


def generate_melody(chord_sequence, key_root=60, mode='major', seed=None):
    """Generate a simple melody based on chord sequence.
    
    Returns:
        List of (note, duration, start_time) tuples
    """
    if seed is not None:
        np.random.seed(seed + 1)  # Different seed from chords
    
    melody = []
    time = 0
    
    for chord_label in chord_sequence:
        # Get notes that work with this chord
        chord_notes = get_chord_notes(chord_label, key_root, mode, octave_offset=1)
        scale_notes = get_chord_scale_notes(chord_label, key_root, mode, octave_offset=1)
        
        # Prefer chord tones (70% probability)
        available_notes = chord_notes + scale_notes
        weights = [0.7/len(chord_notes)] * len(chord_notes) + [0.3/len(scale_notes)] * len(scale_notes)
        
        # Generate notes for this chord (2 beats total)
        chord_duration = 2.0
        current_time = time
        
        # Random rhythm: quarter notes, eighth notes, or half notes
        rhythm_choice = np.random.choice(['quarter', 'eighth', 'half'])
        
        if rhythm_choice == 'quarter':
            # Two quarter notes
            for _ in range(2):
                note = np.random.choice(available_notes, p=weights)
                melody.append((note, 1.0, current_time))
                current_time += 1.0
        elif rhythm_choice == 'eighth':
            # Four eighth notes
            for _ in range(4):
                note = np.random.choice(available_notes, p=weights)
                melody.append((note, 0.5, current_time))
                current_time += 0.5
        else:
            # One half note
            note = np.random.choice(available_notes, p=weights)
            melody.append((note, 2.0, current_time))
            current_time += 2.0
        
        time += chord_duration
    
    return melody


def create_full_midi(chord_sequence, output_path, mode='major', key_root=60, tempo=120, 
                     add_melody=True, add_bass=True, seed=None):
    """Create MIDI file with chords, melody, and bass.
    
    Args:
        chord_sequence: List of functional chord labels
        output_path: Output MIDI file path
        mode: 'major' or 'minor'
        key_root: MIDI note for key root (60=C, 62=D, 64=E, 65=F, 67=G, 69=A, 71=B)
        tempo: BPM
        add_melody: Whether to add melody track
        add_bass: Whether to add bass track
        seed: Random seed
    """
    # Create MIDI file with multiple tracks
    num_tracks = 1 + (1 if add_melody else 0) + (1 if add_bass else 0)
    midi = MIDIFile(num_tracks)
    
    track_idx = 0
    
    # Track 0: Chords (Piano)
    chord_track = track_idx
    midi.addTempo(chord_track, 0, tempo)
    midi.addProgramChange(chord_track, 0, 0, 0)  # Piano
    
    time = 0
    duration = 2
    volume = 80
    
    for chord_label in chord_sequence:
        notes = get_chord_notes(chord_label, key_root, mode)
        for note in notes:
            midi.addNote(chord_track, 0, note, time, duration, volume)
        time += duration
    
    track_idx += 1
    
    # Track 1: Melody (Lead)
    if add_melody:
        melody_track = track_idx
        midi.addProgramChange(melody_track, 1, 0, 0)  # Piano (could change to other instruments)
        
        melody_notes = generate_melody(chord_sequence, key_root, mode, seed)
        for note, dur, start_time in melody_notes:
            midi.addNote(melody_track, 1, note, start_time, dur, 90)
        
        track_idx += 1
    
    # Track 2: Bass (Bass line)
    if add_bass:
        bass_track = track_idx
        midi.addProgramChange(bass_track, 2, 0, 32)  # Acoustic Bass
        
        time = 0
        duration = 2
        for chord_label in chord_sequence:
            # Bass plays root note, 2 octaves below
            chord_notes = get_chord_notes(chord_label, key_root, mode, octave_offset=-2)
            bass_note = chord_notes[0]  # Root note
            midi.addNote(bass_track, 2, bass_note, time, duration, 90)
            time += duration
    
    # Write MIDI file
    with open(output_path, 'wb') as f:
        midi.writeFile(f)
    
    print(f"✅ Full MIDI file created: {output_path}")
    print(f"   - {len(chord_sequence)} chords")
    print(f"   - Tracks: Chords" + (" + Melody" if add_melody else "") + (" + Bass" if add_bass else ""))
    print(f"   - Duration: {len(chord_sequence) * 2} beats (~{len(chord_sequence) * 2 / tempo * 60:.1f}s at {tempo} BPM)")


def main():
    parser = argparse.ArgumentParser(description="Generate full music with melody and harmony")
    parser.add_argument('--model', type=str, required=True, help='Path to trained HMM model (.pkl)')
    parser.add_argument('--mode', type=str, default='major', choices=['major', 'minor'])
    parser.add_argument('--bars', type=int, default=8, help='Number of bars (4 beats per bar, 2 chords per bar)')
    parser.add_argument('--output', type=str, default='generated_full_music.mid')
    parser.add_argument('--key', type=int, default=60, help='Key root (60=C, 62=D, 64=E, 65=F, 67=G, 69=A, 71=B)')
    parser.add_argument('--tempo', type=int, default=120, help='Tempo in BPM')
    parser.add_argument('--melody', action='store_true', default=True, help='Add melody track')
    parser.add_argument('--no-melody', dest='melody', action='store_false', help='No melody track')
    parser.add_argument('--bass', action='store_true', default=True, help='Add bass track')
    parser.add_argument('--no-bass', dest='bass', action='store_false', help='No bass track')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    # Calculate number of chords (2 chords per bar, 4 beats per bar)
    num_chords = args.bars * 2
    
    # Load model
    print(f"📂 Loading model: {args.model}")
    with open(args.model, 'rb') as f:
        params = pickle.load(f)
    print(f"✅ Model loaded (vocabulary: {len(params['vocab'])} chords)")
    
    # Generate chord sequence
    print(f"\n🎵 Generating {num_chords} chords ({args.bars} bars) in {args.mode} mode...")
    chord_sequence = generate_chord_sequence(params, mode=args.mode, length=num_chords, seed=args.seed)
    
    print(f"✅ Chord progression:")
    # Print chords in groups of 4 (2 bars)
    for i in range(0, len(chord_sequence), 4):
        bar_chords = chord_sequence[i:i+4]
        print(f"   Bars {i//4 + 1}-{i//4 + 2}: {' - '.join(bar_chords)}")
    
    # Create MIDI
    print(f"\n🎹 Creating MIDI file...")
    create_full_midi(chord_sequence, args.output, mode=args.mode, key_root=args.key, 
                     tempo=args.tempo, add_melody=args.melody, add_bass=args.bass, seed=args.seed)
    
    key_names = {60: 'C', 62: 'D', 64: 'E', 65: 'F', 67: 'G', 69: 'A', 71: 'B'}
    key_name = key_names.get(args.key, f'MIDI_{args.key}')
    
    print(f"\n✨ Done!")
    print(f"   Key: {key_name} {args.mode}")
    print(f"   Tempo: {args.tempo} BPM")
    print(f"\n   📱 How to listen:")
    print(f"   1. Open '{args.output}' in GarageBand")
    print(f"   2. Use QuickTime Player (File > Open)")
    print(f"   3. Use any online MIDI player")


if __name__ == '__main__':
    main()
