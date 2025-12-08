"""Generate FULL music using HDP-HSMM.
Leverages HSMM durations for dynamic accompaniment patterns.

Location: music_generation/generate_hdp_hsmm_full.py
Usage:
  python music_generation/generate_hdp_hsmm_full.py --model models/hdp_hsmm.pkl --bars 16 --output hdp_hsmm_full.mid
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

def get_chord_notes(label, key_root=60):
    # (Same helper as above, simplified)
    if label == "N": return []
    parts = label.split(':')
    base = parts[0].replace('b','').replace('#','')
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    roman_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}
    degree = roman_map.get(base.upper(), 0)
    root = key_root + major_scale[degree]
    is_minor = 'min' in label or base.islower()
    return [root, root+(3 if is_minor else 4), root+7]

def create_full_arrangement(indices, output_path, inv_vocab, key_root=60):
    midi = MIDIFile(3)
    midi.addTempo(0, 0, 110)
    midi.addProgramChange(0, 0, 0, 0) # Piano
    midi.addProgramChange(1, 1, 73, 0) # Flute/Lead
    midi.addProgramChange(2, 2, 32, 0) # Bass

    # Group beats into (chord_idx, duration) segments
    segments = []
    if not indices: return
    curr = indices[0]
    dur = 0
    for idx in indices:
        if idx == curr:
            dur += 1
        else:
            segments.append((curr, dur))
            curr = idx
            dur = 1
    segments.append((curr, dur))

    time = 0
    for idx, duration in segments:
        label = inv_vocab.get(idx, "N")
        if label == "N": 
            time += duration
            continue

        notes = get_chord_notes(label, key_root)
        
        # --- 1. Smart Accompaniment based on Duration ---
        if duration >= 4:
            # Long chord: Arpeggio
            for i in range(int(duration * 2)): # 8th notes
                note = notes[i % 3]
                midi.addNote(0, 0, note, time + i*0.5, 0.5, 80)
        else:
            # Short chord: Block chords
            for n in notes:
                midi.addNote(0, 0, n, time, duration, 70)

        # --- 2. Bass ---
        # Plays root on beat 1, fifth on beat 3 (if long enough)
        midi.addNote(2, 2, notes[0]-12, time, 2, 90)
        if duration > 2:
            midi.addNote(2, 2, notes[2]-12, time+2, duration-2, 85)

        # --- 3. Melody ---
        # Generate a simple scale run or sustained note
        lead_note = notes[0] + 12 # Octave up
        midi.addNote(1, 1, lead_note, time, duration, 95)
        
        time += duration

    with open(output_path, "wb") as f:
        midi.writeFile(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/hdp_hsmm.pkl')
    parser.add_argument('--bars', type=int, default=16)
    parser.add_argument('--output', default='hdp_hsmm_full.mid')
    args = parser.parse_args()

    # --- Path Correction Logic for Model ---
    model_path = Path(args.model)
    if not model_path.exists():
        # Try finding relative to project root
        model_path = PROJECT_ROOT / args.model
    
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {model_path}")
        return

    # --- 📂 Output Path Logic ---
    # 1. 定义目标文件夹: Project_Root/generated_music
    output_dir = PROJECT_ROOT / "generated_music"
    
    # 2. 自动创建文件夹 (如果不存在)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 组合完整路径 (只取文件名)
    output_filename = Path(args.output).name
    final_output_path = output_dir / output_filename

    print(f"Loading HDP-HSMM: {model_path}")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    model = data['model']
    inv_vocab = {v: k for k, v in data['vocab'].items()}
    
    total_beats = args.bars * 4
    print(f"Generating {total_beats} beats...")
    
    indices = model.generate(melody_length=total_beats)
    
    create_full_arrangement(indices, final_output_path, inv_vocab)
    print(f"✅ Full HSMM arrangement saved to: {final_output_path}")

if __name__ == "__main__":
    main()