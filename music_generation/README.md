# 🎵 Music Generation Scripts

Tools for generating music using trained HMM models.

## 📂 Files

### Scripts
- **`generate_full_music.py`** ⭐ Main tool - Generate complete music (chords + melody + bass)
- **`generate_music.py`** - Generate chord progressions only
- **`quick_test.py`** - Quick test utility

### Documentation
- **`MUSIC_QUICKSTART.md`** ⭐ Quick reference
- **`MUSIC_GENERATION_GUIDE.md`** - Detailed usage guide

## 🚀 Quick Start

### Generate Complete Music
```bash
python music_generation/generate_full_music.py \
    --model models/hmm_conditional.pkl \
    --mode major \
    --bars 8 \
    --output generated_music/my_song.mid
```

### Quick Test
```bash
python music_generation/quick_test.py
```

## 📝 Common Commands

### Pop Song Style (C Major)
```bash
python music_generation/generate_full_music.py \
    --model models/hmm_conditional.pkl \
    --mode major \
    --bars 16 \
    --key 60 \
    --tempo 120 \
    --output generated_music/pop.mid
```

### Ballad Style (A Minor)
```bash
python music_generation/generate_full_music.py \
    --model models/hmm_conditional.pkl \
    --mode minor \
    --bars 12 \
    --key 69 \
    --tempo 75 \
    --output generated_music/ballad.mid
```

### Batch Generation
```bash
for i in {1..5}; do
    python music_generation/generate_full_music.py \
        --model models/hmm_conditional.pkl \
        --mode major \
        --bars 8 \
        --output generated_music/song_$i.mid \
        --seed $i
done
```

## 🎧 Play Generated Music

```bash
# Play a single file
open generated_music/my_song.mid

# Open in GarageBand
open -a GarageBand generated_music/my_song.mid

# View all generated music
ls -lh generated_music/
```

## 📖 More Information

See detailed documentation:
- Quick reference: `MUSIC_QUICKSTART.md`
- Complete guide: `MUSIC_GENERATION_GUIDE.md`
