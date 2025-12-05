# Bayesian Chord Progression Model

A Bayesian approach to modeling chord progressions in pop music using Hidden Markov Models (HMM) with functional harmony and mode-conditional modeling.

## Overview

This project implements and compares two HMM-based approaches for chord progression modeling:
1. **Baseline Model**: Absolute chord representation (193 unique chords)
2. **Conditional Model**: Functional harmony with mode conditioning (20 functional chords)

**Key Results:**
- Conditional model achieves **56.6% perplexity reduction** over baseline (4.96 vs 11.44)
- **41.82% prediction accuracy** (+23.0% improvement over baseline)
- **89.6% vocabulary reduction** through functional chord representation (193 → 20 chords)

## Dataset
- **POP909**: 909 pop songs with melody-chord-piano alignment ([GitHub](https://github.com/music-x-lab/POP909-Dataset))
- **Split**: 70% train (636 songs) / 15% validation (136 songs) / 15% test (137 songs)
- **Features**: Beat-aligned chord annotations, key signatures, mode labels

## Models

### Baseline HMM (`hmm_baseline.pkl`)
- Absolute chord representation (C, C#m, D7, etc.)
- 193 unique chords after C/Am transposition
- Trained on 636 songs (70% of dataset)
- Test perplexity: 11.44
- Test accuracy: 33.99%

### Conditional HMM (`hmm_conditional.pkl`) ⭐ Best Model
- Functional harmony (Roman numerals: I, IV, V, vi, etc.)
- Mode-conditional: P(chord | previous_chord, mode) where mode ∈ {major, minor}
- 20 functional chords (7 major + 7 minor diatonic + 5 chromatic + 1 special)
- Test perplexity: 4.96 (56.6% improvement over baseline)
- Test accuracy: 41.82% (23.0% improvement over baseline)
- Handles chromatic chords: bII (Neapolitan), bVII (subtonic), #IV, etc.


## Project Structure
```
.
├── data/
│   ├── POP909/              # POP909 dataset (909 songs)
│   └── archive/             # Classical music dataset (optional)
├── models/
│   ├── hmm_baseline.pkl     # Baseline model (193 chords, trained on 636 songs)
│   └── hmm_conditional.pkl  # Conditional model (20 functional chords) ⭐
├── scripts/
│   ├── train_baseline.py       # Train baseline model
│   ├── train_conditional.py    # Train conditional model
│   ├── test_baseline.py        # Test baseline model
│   ├── test_conditional.py     # Test conditional model
│   ├── compare_models.py       # Compare model performance
│   └── update_model_accuracy.py # Update model metadata with test accuracy
├── notebooks/
│   ├── 01_EDA_Music_Datasets.ipynb # Dataset exploration
│   └── Model_Visualization.ipynb   # Model visualization & analysis
├── validation_results/
│   ├── BASELINE_SUMMARY.md         # Baseline model report
│   └── CONDITIONAL_SUMMARY.md      # Conditional model report ⭐
├── src/
│   └── data/                # Dataset loaders & preprocessing
│       ├── pop909_parser.py      # POP909 dataset loader
│       ├── chord_preprocessing.py # Chord normalization & transposition
│       └── key_aware_features.py  # Functional chord extraction
└── README.md
```



## Quick Start

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Download Dataset
Download POP909 dataset and place it in `data/POP909/`

### 3. Train Models

**Train baseline model:**
```bash
python scripts/train_baseline.py --pop909 data/POP909 --out models/hmm_baseline.pkl
```

**Train conditional model (recommended):**
```bash
python scripts/train_conditional.py --pop909 data/POP909 --out models/hmm_conditional.pkl
```

Add `--limit 20` for quick testing on 20 songs.

### 4. Evaluate Models

**Test baseline:**
```bash
python scripts/test_baseline.py --model models/hmm_baseline.pkl --pop909 data/POP909
```

**Test conditional model:**
```bash
python scripts/test_conditional.py --model models/hmm_conditional.pkl --pop909 data/POP909
```

**Compare models:**
```bash
python scripts/compare_models.py
```

### 5. Generate Music 🎵 NEW!
Use your trained model to generate new chord progressions and convert them to playable MIDI:

**Quick test (generates test song):**
```bash
python music_generation/quick_test.py
```

**Generate full music with melody and bass:**
```bash
python music_generation/generate_full_music.py \
    --model models/hmm_conditional.pkl \
    --mode major \
    --bars 8 \
    --output generated_music/my_song.mid
```

**Generate simple chord progression:**
```bash
python music_generation/generate_music.py \
    --model models/hmm_conditional.pkl \
    --mode major \
    --length 32 \
    --output generated_music/chords.mid
```

**Play the generated MIDI:**
- Open `.mid` file in GarageBand (macOS)
- Use QuickTime Player: `open generated_music/my_song.mid`
- Upload to online MIDI player

📖 **See [`music_generation/`](music_generation/) folder for detailed instructions!**

### 6. Visualize Results
Open `notebooks/Model_Visualization.ipynb` to visualize:
- Model performance comparison
- Transition probability heatmaps
- Chord distribution analysis
- Mode-specific patterns

## Results Summary

See detailed reports:
- [`validation_results/BASELINE_SUMMARY.md`](validation_results/BASELINE_SUMMARY.md) - Baseline model analysis
- [`validation_results/CONDITIONAL_SUMMARY.md`](validation_results/CONDITIONAL_SUMMARY.md) - Conditional model analysis ⭐

**Key Findings:**
- Functional harmony dramatically reduces vocabulary (193 → 20 chords, 89.6% reduction)
- Mode-conditional modeling captures major/minor differences in progression patterns
- 56.6% perplexity improvement (11.44 → 4.96)
- 23.0% accuracy improvement (33.99% → 41.82%, +1755 correct predictions)
- Chromatic chords (bII, bVII) are frequent in pop music and handled correctly
- Prediction errors are musically coherent (functionally similar substitutions)

## Technical Details

**Modeling Approach:**
- 1st-order Markov model: P(chord_t | chord_{t-1}, mode)
- Add-one smoothing for unseen transitions
- Key transposition to C major / A minor for data pooling
- Functional chord extraction using scale degree analysis

**Evaluation Metrics:**
- **Perplexity**: Measures model uncertainty (lower = better)
- **Prediction Accuracy**: Next-chord prediction correctness
- **Vocabulary Coverage**: Percentage of chords in test data seen during training

## Future Work
- 2nd-order Markov models (bigram → trigram)
- Melody conditioning: P(chord | previous_chord, melody, mode)
- Rhythm and duration modeling
- Neural baselines (LSTM/Transformer) for comparison
