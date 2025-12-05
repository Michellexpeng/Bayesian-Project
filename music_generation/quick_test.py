#!/usr/bin/env python3
"""Quick test script to generate a sample song.

快速测试脚本 - 生成一首示例歌曲来验证系统正常工作。
"""
from __future__ import annotations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("🎵 Bayesian Music Generator - Quick Test")
    print("=" * 50)
    
    # Check if model exists
    model_path = PROJECT_ROOT / "models" / "hmm_conditional.pkl"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first:")
        print("   python scripts/train_conditional.py --pop909 data/POP909 --out models/hmm_conditional.pkl")
        return 1
    
    print(f"✅ Found model: {model_path}")
    
    # Try to import required libraries
    try:
        from midiutil import MIDIFile
        print("✅ MIDIUtil installed")
    except ImportError:
        print("❌ MIDIUtil not installed")
        print("   Install it with: pip install MIDIUtil")
        return 1
    
    try:
        import numpy as np
        import pickle
        print("✅ NumPy and pickle available")
    except ImportError as e:
        print(f"❌ Required library missing: {e}")
        return 1
    
    # Generate a test song
    print("\n🎹 Generating test song...")
    output_path = PROJECT_ROOT / "generated_music" / "test_song.mid"
    
    # Import the generation script
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    
    try:
        from generate_full_music import generate_chord_sequence, create_full_midi
        
        # Load model
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        
        # Generate 8 bars (16 chords) in C major
        chord_sequence = generate_chord_sequence(params, mode='major', length=16, seed=42)
        
        print(f"✅ Generated chord sequence:")
        print(f"   {' - '.join(chord_sequence[:8])}")
        print(f"   {' - '.join(chord_sequence[8:])}")
        
        # Create MIDI file
        create_full_midi(chord_sequence, str(output_path), mode='major', 
                        key_root=60, tempo=110, add_melody=True, add_bass=True, seed=42)
        
        print(f"\n✨ Success! Test song created: {output_path}")
        print(f"\n📱 How to listen:")
        print(f"   • Open in GarageBand: open {output_path}")
        print(f"   • Use QuickTime Player")
        print(f"   • Upload to online MIDI player")
        
        print(f"\n🚀 Next steps:")
        print(f"   • Read MUSIC_GENERATION_GUIDE.md for more options")
        print(f"   • Try: python scripts/generate_full_music.py --mode minor --bars 8")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
