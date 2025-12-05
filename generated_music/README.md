# 🎵 Generated Music

This folder contains MIDI music files generated using the HMM model.

## 📂 Current Files

Generated MIDI files are saved here.

## 🎧 How to Play

### macOS
```bash
# Open with default application
open filename.mid

# Open with GarageBand
open -a GarageBand filename.mid

# Open with QuickTime Player
open -a "QuickTime Player" filename.mid
```

### Batch Playback
```bash
# Play all generated music
for file in *.mid; do
    echo "Playing: $file"
    open "$file"
    sleep 5
done
```

## 🎹 Edit and Export

1. Open MIDI file in GarageBand
2. You can:
   - Change instrument sounds
   - Adjust volume and effects
   - Add drums and other tracks
   - Export as MP3/AAC audio formats

## 🔄 Regenerate

To generate new music, use:
```bash
python music_generation/generate_full_music.py \
    --model models/hmm_conditional.pkl \
    --mode major \
    --bars 8 \
    --output generated_music/new_song.mid
```

See `music_generation/README.md` for more generation options.
