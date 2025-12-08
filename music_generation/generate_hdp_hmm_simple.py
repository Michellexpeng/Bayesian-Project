"""Generate simple chord progression using HDP-HMM.

Usage:
  python music_generation/generate_hdp_hmm_simple.py --model models/hdp_hmm.pkl --length 32 --output hdp_hmm_simple.mid
"""
import argparse
import pickle
import sys
import numpy as np
from pathlib import Path
from midiutil import MIDIFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hdp_hmm import KeyAwareHDPHMM

def roman_to_midi_notes(roman_numeral, key_root=60):
    """Simple Roman to MIDI mapping (Major Scale default for simplicity)."""
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    roman_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}
    
    # Parse Label
    clean_label = roman_numeral.split(':')[0].replace('b', '').replace('#', '')
    alteration = -1 if 'b' in roman_numeral.split(':')[0] else (1 if '#' in roman_numeral else 0)
    
    if clean_label not in roman_map: return [] # Skip N or unknown
    
    degree = roman_map.get(clean_label, 0)
    root = key_root + major_scale[degree] + alteration
    
    # Simple Triad: Root, 3rd, 5th
    is_minor = roman_numeral.lower() == roman_numeral.split(':')[0]
    third = root + (3 if is_minor else 4)
    fifth = root + 7
    return [root, third, fifth]

def create_midi(sequence, output_path, vocab_inv, tempo=120):
    midi = MIDIFile(1)
    track = 0
    time = 0
    duration = 1.0 # Assume model generated beats
    
    midi.addTempo(track, time, tempo)
    
    for idx in sequence:
        label = vocab_inv.get(idx, "N")
        if label != "N":
            notes = roman_to_midi_notes(label)
            for note in notes:
                midi.addNote(track, 0, note, time, duration, 100)
        time += duration
        
    with open(output_path, "wb") as f:
        midi.writeFile(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/hdp_hmm.pkl')
    parser.add_argument('--length', type=int, default=32)
    parser.add_argument('--output', default='hdp_hmm_simple.mid')
    args = parser.parse_args()


    model_path = Path(args.model)
    if not model_path.exists():
        model_path = PROJECT_ROOT / args.model
        
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {model_path}")
        return

    # --- 3. 修改点：强制保存到 generated_music 文件夹 ---
    output_dir = PROJECT_ROOT / "generated_music"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 只取文件名，忽略用户路径中的文件夹部分
    output_filename = Path(args.output).name
    final_output_path = output_dir / output_filename

    print(f"Loading HDP-HMM: {model_path}")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    model = data['model']
    inv_vocab = {v: k for k, v in data['vocab'].items()}
    
    print(f"Generating {args.length} beats...")
    # HMM generation gives indices per step
    indices = model.generate(length=args.length)
    
    labels = [inv_vocab.get(i, "N") for i in indices]
    print(f"Sequence: {' -> '.join(labels[:10])}...")
    
    create_midi(indices, final_output_path, inv_vocab)
    print(f"✅ Saved to {final_output_path}")

if __name__ == "__main__":
    main()