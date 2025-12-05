# 🎵 Music Generation - Quick Reference

## ✅ What You Can Do Now

Your HMM model **can predict the next chord**. Now you can:

1. ✅ **Generate new chord sequences** (using trained model)
2. ✅ **Convert to MIDI files** (playable in GarageBand/QuickTime)
3. ✅ **Add melody and bass** (complete music experience)

## 🚀 Three Ways to Generate Music

### Method 1: Quick Test (Easiest)
```bash
python music_generation/quick_test.py
```
Generates `test_song.mid` and tells you how to play it

### Method 2: Complete Music (Recommended)
```bash
python music_generation/generate_full_music.py \
    --model models/hmm_conditional.pkl \
    --mode major \
    --bars 8 \
    --output generated_music/my_song.mid
```
Generates complete music with melody and bass

### Method 3: Chords Only
```bash
python music_generation/generate_music.py \
    --model models/hmm_conditional.pkl \
    --mode major \
    --length 32 \
    --output generated_music/chords.mid
```
Generates chord progression only, no melody

## 🎧 How to Play

### macOS (Easiest)
```bash
open generated_music/my_song.mid
```
Automatically opens with QuickTime Player or GarageBand

### GarageBand (Best Sound Quality)
1. Open GarageBand
2. Drag and drop `.mid` file
3. You can:
   - Change instrument sounds
   - Adjust volume
   - Export to MP3

### Online Players
Upload to:
- https://onlinesequencer.net/import
- https://signal.vercel.app/edit

## 📝 Sample Files Already Generated

I've generated 3 examples for you:

| File | Style | Key | Tempo | Description |
|------|-------|-----|-------|-------------|
| `test_song.mid` | Test | C major | 110 BPM | Quick test generation |
| `happy_song.mid` | Happy | G major | 120 BPM | Pop song style |
| `sad_song.mid` | Ballad | A minor | 80 BPM | Slow minor key |

**Play them now:**
```bash
open generated_music/test_song.mid
open generated_music/happy_song.mid  
open generated_music/sad_song.mid
```

## 🎼 Parameter Reference

### Mode
- `major` = Major key (sounds happy)
- `minor` = Minor key (sounds sad)

### Key - MIDI Note Numbers
| Note | MIDI | Usage Example |
|------|------|---------------|
| C | 60 | `--key 60` |
| D | 62 | `--key 62` |
| E | 64 | `--key 64` |
| F | 65 | `--key 65` |
| G | 67 | `--key 67` |
| A | 69 | `--key 69` |
| B | 71 | `--key 71` |

### Tempo
- Slow: 60-80 BPM (ballad)
- Medium: 90-120 BPM (pop)
- Fast: 130-160 BPM (dance)

### Bars (Measures)
- 4 bars = 8 chords = ~8 seconds
- 8 bars = 16 chords = ~16 seconds
- 16 bars = 32 chords = ~32 seconds

## 💡 Practical Examples

### Generate Pop Song Style
```bash
python music_generation/generate_full_music.py \
    --model models/hmm_conditional.pkl \
    --mode major \
    --bars 16 \
    --key 60 \
    --tempo 120 \
    --output generated_music/pop.mid
```

### Generate Ballad Style
```bash
python music_generation/generate_full_music.py \
    --model models/hmm_conditional.pkl \
    --mode minor \
    --bars 12 \
    --key 69 \
    --tempo 75 \
    --output generated_music/ballad.mid
```

### Batch Generate 10 Songs
```bash
for i in {1..10}; do
    python music_generation/generate_full_music.py \
        --model models/hmm_conditional.pkl \
        --mode major \
        --bars 8 \
        --output generated_music/song_$i.mid \
        --seed $i
done
```

## 📚 More Information

Detailed documentation: [`MUSIC_GENERATION_GUIDE.md`](MUSIC_GENERATION_GUIDE.md)

## 🎯 How It Works

```
HMM Model → Predict Next Chord → Generate Chord Sequence → Convert to MIDI Notes → MIDI File
```

Your model learned chord transition probabilities from 909 pop songs:
- P(next chord | current chord, mode)
- Uses functional harmony (I, IV, V, vi, etc.)
- Distinguishes major and minor mode patterns

## ❓ FAQ

**Q: How does the music sound?**
- Chord progressions follow pop music patterns
- Melodies are randomly generated, relatively simple
- Can be further edited in GarageBand

**Q: How to generate better music?**
- Try different seed values: `--seed 1`, `--seed 2`, ...
- Use conditional model (already using it)
- Generate multiple versions, pick the best
- Edit and refine in GarageBand

**Q: Can I use original dataset MIDI?**
```bash
# Play original POP909 data
open data/POP909/001/*.mid
```

**Q: Want to export MP3?**
In GarageBand: File → Share → Export Song to Disk → Select MP3 format
