"""Generate FULL music (Chords + Melody + Bass) using HDP-HMM.

Location: music_generation/generate_hdp_hmm_full.py
Usage:
  python music_generation/generate_hdp_hmm_full.py --model models/hdp_hmm.pkl --bars 8 --output hdp_hmm_full.mid
"""
import argparse
import pickle
import sys
import numpy as np
from pathlib import Path
from midiutil import MIDIFile

# --- 1. Path Setup ---
# Calculates project root correctly whether run from root or subfolder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import full helper logic
from src.models.hdp_hmm import KeyAwareHDPHMM

# --- Reusing Logic (Adapted) ---
def get_chord_notes(roman_numeral, key_root=60):
    roman_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}
    parts = roman_numeral.split(':')
    base = parts[0]
    quality = parts[1] if len(parts)>1 else 'maj'
    
    alteration = 0
    if base.startswith('b'): alteration = -1; base = base[1:]
    elif base.startswith('#'): alteration = 1; base = base[1:]
    
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    degree = roman_map.get(base.upper(), 0)
    root = key_root + major_scale[degree] + alteration
    
    is_minor = 'min' in quality or base.islower()
    third = root + (3 if is_minor else 4)
    fifth = root + 7
    return [root, third, fifth]

def generate_melody_for_beat(chord_label, key_root):
    """Generate 1 beat of melody."""
    chord_notes = get_chord_notes(chord_label, key_root)
    return np.random.choice(chord_notes + [n+12 for n in chord_notes])

def create_full_midi(sequence, output_path, inv_vocab, key_root=60):
    midi = MIDIFile(3) # Tracks: 0=Chords, 1=Melody, 2=Bass
    midi.addTempo(0, 0, 120)
    
    # Instruments
    midi.addProgramChange(0, 0, 0, 0)  # Piano (Chords)
    midi.addProgramChange(1, 1, 0, 0)  # Piano (Melody)
    midi.addProgramChange(2, 2, 32, 0) # Acoustic Bass
    
    time = 0
    duration = 1.0 # 1 beat per HMM step
    
    for idx in sequence:
        label = inv_vocab.get(idx, "N")
        if label == "N": 
            time += duration
            continue
            
        # 1. Chords
        notes = get_chord_notes(label, key_root)
        for n in notes:
            midi.addNote(0, 0, n, time, duration, 75)
            
        # 2. Bass (Root note dropped 1 octave)
        midi.addNote(2, 2, notes[0]-12, time, duration, 90)
        
        # 3. Melody (Simple 8th notes)
        m_note1 = generate_melody_for_beat(label, key_root)
        m_note2 = generate_melody_for_beat(label, key_root)
        midi.addNote(1, 1, m_note1, time, 0.5, 95)
        midi.addNote(1, 1, m_note2, time+0.5, 0.5, 95)
        
        time += duration
        
    with open(output_path, "wb") as f:
        midi.writeFile(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/hdp_hmm.pkl')
    parser.add_argument('--bars', type=int, default=8)
    parser.add_argument('--output', default='hdp_hmm_full.mid', help="Filename of the output MIDI")
    args = parser.parse_args()

    # --- Path Correction Logic for Model ---
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = PROJECT_ROOT / args.model
    
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {model_path}")
        return

    output_dir = PROJECT_ROOT / "generated_music"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = Path(args.output).name
    final_output_path = output_dir / output_filename

    print(f"Loading HDP-HMM: {model_path}")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    model = data['model']
    inv_vocab = {v: k for k, v in data['vocab'].items()}
    
    # 4 beats per bar
    total_beats = args.bars * 4
    print(f"Generating {total_beats} beats...")
    
    indices = model.generate(length=total_beats)
    
    create_full_midi(indices, final_output_path, inv_vocab)
    print(f"✅ Full arrangement saved to: {final_output_path}")

if __name__ == "__main__":
    main()