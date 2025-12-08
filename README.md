# Bayesian Chord Progression Model

A Bayesian approach to modeling chord progressions in pop music using Hidden Markov Models (HMM) with functional harmony and mode-conditional modeling.

## Overview

This project implements and compares two HMM-based approaches for chord progression modeling:
1. **Baseline Model**: Absolute chord representation (193 unique chords)
2. **Conditional Model**: Functional harmony with mode conditioning (20 functional chords)
3. **HDP-HMM**: Bayesian Nonparametric HMM with infinite state space and key-aware priors.
4. **HDP-HSMM**: Hierarchical Dirichlet Process Hidden Semi-Markov Model with explicit duration modeling.

**Key Results:**
- Conditional model achieves **56.6% perplexity reduction** over baseline (4.96 vs 11.44)
- **41.82% prediction accuracy** (+23.0% improvement over baseline)
- **89.6% vocabulary reduction** through functional chord representation (193 → 20 chords)
- HDP-HSMM outperforms HDP-HMM in both perplexity (8.75 vs 9.11) and accuracy (48.08% vs 46.16%), demonstrating the value of explicit duration modeling for musical structure.
- Functional Harmony reduces vocabulary size by 89.6% (193 $\to$ 20 chords), enabling more robust learning.

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

### Conditional HMM (`hmm_conditional.pkl`) 
- Functional harmony (Roman numerals: I, IV, V, vi, etc.)
- Mode-conditional: P(chord | previous_chord, mode) where mode ∈ {major, minor}
- 20 functional chords (7 major + 7 minor diatonic + 5 chromatic + 1 special)
- Test perplexity: 4.96 (56.6% improvement over baseline)
- Test accuracy: 41.82% (23.0% improvement over baseline)
- Handles chromatic chords: bII (Neapolitan), bVII (subtonic), #IV, etc.

### HDP-HMM (`hdp_hmm.pkl`)
- Type: Hierarchical Dirichlet Process HMM
- Features: Functional harmony + Key-Aware Priors
- Mechanism: Automatically infers the number of hidden harmonic states (infinite state space).
- Performance: Perplexity 9.11, Accuracy 46.16%

### HDP-HSMM (`hdp_hsmm.pkl`) ⭐ Highest Accuracy
- Type: HDP Hidden Semi-Markov Model
- Features: Functional harmony + Key-Aware Priors + Explicit Duration Modeling
- Mechanism: Models state duration explicitly (e.g., how long a chord lasts), solving the "rapid switching" problem of standard HMMs.
- Performance: Perplexity 8.75, Accuracy 48.08%

## Project Structure
```
.
├── data/
│   ├── POP909/              # POP909 dataset root
├── models/
│   ├── hmm_baseline.pkl
│   ├── hmm_conditional.pkl
│   ├── hdp_hmm.pkl          # HDP-HMM model
│   └── hdp_hsmm.pkl         # HDP-HSMM model
├── generated_music/         # Output folder for MIDI files
├── scripts/
│   ├── train_baseline.py
│   ├── train_conditional.py
│   ├── train_hdp_hmm.py     # Train HDP-HMM
│   ├── train_hdp_hsmm.py    # Train HDP-HSMM
│   ├── test_baseline.py
│   ├── test_conditional.py
│   ├── test_hdp_hmm.py      # Test HDP-HMM
│   ├── test_hdp_hsmm.py     # Test HDP-HSMM
│   ├── compare_models.py    # Compare Baseline vs Conditional
│   └── compare_hdp_models.py # Compare HDP-HMM vs HDP-HSMM
├── music_generation/        # Generation scripts
│   ├── generate_hdp_hmm_simple.py
│   ├── generate_hdp_hmm_full.py
│   ├── generate_hdp_hsmm_simple.py
│   └── generate_hdp_hsmm_full.py
├── src/
│   ├── data/                # Data processing modules
│   └── models/              # Core model classes
│       ├── hdp_hmm.py       # HDP-HMM implementation
│       └── hdp_hsmm.py      # HDP-HSMM implementation
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

**Train conditional model:**
```bash
python scripts/train_conditional.py --pop909 data/POP909 --out models/hmm_conditional.pkl
```

Add `--limit 20` for quick testing on 20 songs.

**Train HDP-HMM:**
```bash
python scripts/train_hdp_hmm.py
```

**Train HDP-HSMM:**
```bash
python scripts/train_hdp_hsmm.py
```

### 4. Evaluate Models

**Test baseline:**
```bash
python scripts/test_baseline.py --model models/hmm_baseline.pkl --pop909 data/POP909
```

**Test conditional model:**
```bash
python scripts/test_conditional.py --model models/hmm_conditional.pkl --pop909 data/POP909
```

**Compare Baseline and Conditional models:**
```bash
python scripts/compare_models.py
```

** Compare HDP-HMM and HDP-HSMM:**
```bash
python scripts/compare_hdp_models.py --pop909 data/POP909
```

### 5. Generate Music 🎵 NEW!
Use your trained model to generate new chord progressions and convert them to playable MIDI:

**Quick test (generates test song):**
```bash
python music_generation/quick_test.py
```

**Generate full music with melody and bass (e.g. conditional):**
```bash
python music_generation/generate_full_music.py \
    --model models/hmm_conditional.pkl \
    --mode major \
    --bars 8 \
    --output generated_music/my_song.mid
```

**Generate simple chord progression (e.g. conditional):**
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
