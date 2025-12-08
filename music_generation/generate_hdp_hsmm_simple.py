"""Generate chord progression using HDP-HSMM (Explicit Duration).

Usage:
  python music_generation/generate_hdp_hsmm_simple.py --model models/hdp_hsmm.pkl --length 32 --output hdp_hsmm_simple.mid
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

from src.models.hdp_hsmm import KeyAwareHDPHSMM

# Reuse roman_to_midi_notes from hmm script (conceptually)
def roman_to_midi_notes(roman_numeral, key_root=60):
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    roman_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}
    clean = roman_numeral.split(':')[0].replace('b','').replace('#','')
    if clean not in roman_map: return []
    
    # Calculate root (Note: this simplified version ignores sharps/flats on root)
    root = key_root + major_scale[roman_map[clean]]
    is_minor = 'min' in roman_numeral.lower() or clean.islower()
    return [root, root + (3 if is_minor else 4), root + 7]

def create_hsmm_midi(indices, output_path, inv_vocab):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)
    
    # HSMM generate() returns a beat-by-beat list (flattened).
    # We want to merge identical consecutive chords to create long MIDI notes
    # instead of many short repeated notes.
    
    time = 0
    current_idx = indices[0]
    current_dur = 0
    
    for idx in indices:
        if idx == current_idx:
            current_dur += 1.0
        else:
            # Write previous chord
            label = inv_vocab.get(current_idx, "N")
            if label != "N":
                notes = roman_to_midi_notes(label)
                for n in notes:
                    midi.addNote(0, 0, n, time, current_dur, 100)
            
            # Advance
            time += current_dur
            current_idx = idx
            current_dur = 1.0
            
    # Write final chord
    label = inv_vocab.get(current_idx, "N")
    if label != "N":
        notes = roman_to_midi_notes(label)
        for n in notes:
            midi.addNote(0, 0, n, time, current_dur, 100)

    with open(output_path, "wb") as f:
        midi.writeFile(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/hdp_hsmm.pkl')
    parser.add_argument('--length', type=int, default=32, help="Total beats")
    parser.add_argument('--output', default='hdp_hsmm_simple.mid')
    args = parser.parse_args()

    # --- 2. Path Correction Logic for Model ---
    model_path = Path(args.model)
    if not model_path.exists():
        # Try finding relative to project root
        model_path = PROJECT_ROOT / args.model
    
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {model_path}")
        return

    # --- 3. Output Path Logic (generated_music) ---
    output_dir = PROJECT_ROOT / "generated_music"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = Path(args.output).name
    final_output_path = output_dir / output_filename

    print(f"Loading HDP-HSMM: {model_path}")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    model = data['model']
    inv_vocab = {v: k for k, v in data['vocab'].items()}
    
    print(f"Generating {args.length} beats with explicit duration...")
    # HSMM.generate automatically handles duration sampling
    indices = model.generate(melody_length=args.length)
    
    create_hsmm_midi(indices, final_output_path, inv_vocab)
    print(f"✅ Saved to {final_output_path}")

if __name__ == "__main__":
    main()