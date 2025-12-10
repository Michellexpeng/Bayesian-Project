# Bayesian & Neural Approaches for Chord Generation Tasks in Pop Music
This project aims to explore how different modeling paradigms — Bayesian models and neural sequence models — capture musical structure in pop music.
We evaluate two categories of tasks:
### Task A — Harmonic Structure Modeling (Bayesian Models)
- Modeling chord progressions using interpretable statistical frameworks (HMM, HDP-HMM, HDP-HSMM), focusing on functional harmony, mode, and explicit duration modeling.
### Task B — Melody → Piano Accompaniment Generation (Neural Models)
- Predicting full piano accompaniment (128-dim piano roll) from melody bars using LSTM encoder-decoder models. This task is significantly more complex and high-dimensional, making it unsuitable for traditional Bayesian models, so we tried to solve it with Neural Models.

#### Why We Use Two Tasks？
Our original goal was to **generate full piano accompaniment from melody tracks**.

However, we found that accompaniment pianorolls are **extremely high-dimensional** (128-dim multi-label states), making them **unsuitable for Bayesian models** such as HMM / HDP-HMM. The state space becomes too large to learn stable transition probabilities.

To keep the Bayesian part meaningful, we redesigned that task to **20-class functional chord prediction**, which fits the assumptions of probabilistic models and allows interpretable harmonic structure learning.

At the same time, we preserved the **original accompaniment-generation task** for the LSTM model, since neural sequence-to-sequence models are capable of handling high-dimensional polyphonic outputs.

Thus our final project consists of:
Bayesian Models → functional harmony modeling (20 chords)
LSTM Model → full pianoroll accompaniment generation (original task)

This setup lets each modeling approach operate where it is most effective, while still addressing the broader theme of learning musical structure.

Below we first present the full results of Task A (Bayesian models), followed by Task B (LSTM accompaniment generation), which extends the project into a more challenging generative scenario.

# Task A: Bayesian Chord Progression Modeling

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

# Task B: Melody → Piano Accompaniment Generation (LSTM Model)

This task aims to generate piano accompaniment directly from melody tracks, using an LSTM auto-regressive decoder architecture. Unlike Task A (Bayesian chord modeling), Task B attempts to model full 128-dimensional polyphonic pianoroll, making it a high-dimensional, multi-label sequence prediction problem.

## Dataset Processing
We use the POP909 dataset, which contains aligned melody, bridge, and piano tracks for 909 pop songs.
Because raw MIDI event timing is inconsistent and difficult for neural networks to learn directly, we convert all tracks into a bar-level pianoroll representation:

### Processing steps
1. Frame Sampling (fs = 2)
- Each beat is sampled into 2 frames (low temporal resolution to reduce model complexity).
- Converts continuous MIDI events into a fixed-length frame sequence.

2. Bar Resampling (steps_per_bar = 16)
- Each bar is normalized to 16 time steps.
- Melody and accompaniment are reshaped into tensors of shape: [num_bars, 16, 128]，where 128 is the pitch dimension.

3. Train/Val/Test Split
- We follow an 80/10/10 split based on songs.
- The split is performed at the song level to prevent data leakage.

4. Model Input / Output Format
For each bar, the dataset provides:
- mel_ctx: melody of previous N bars
- piano_ctx: accompaniment of previous N bars
- mel_bar: melody of the current bar
- piano_bar: ground truth accompaniment of the current bar

## Model Architecture

We adopt a Bar-level Encoder–Decoder LSTM capable of generating a full bar of accompaniment conditioned on melody:

### Encoder

The encoder LSTM takes the concatenation of:
Melody context (ctx_bars × 16 × 128)
Piano context (ctx_bars × 16 × 128)

Encoded using a multi-layer LSTM to obtain a hidden state summarizing the harmonic/melodic history. And it outputs hidden states (h, c).

### Decoder

The decoder generates the accompaniment for the current bar step-by-step.
At each timestep t: input_t = concat(mel_bar[t], prev_piano)
mel_bar[t] is the melody at timestep t

prev_piano is either
 - ground-truth (teacher forcing during training)
 - or previous prediction (during inference)

The decoder uses the encoder’s (h, c) as initialization, so:
The generated accompaniment is conditioned on both music history and the current bar’s melody.

### Output Layer

A linear projection maps LSTM outputs to a 128-dim pitch vector:
logits → 128 probabilities (one per pitch)

A sigmoid is applied later during inference; thresholding + top-K selection controls sparsity.

### Loss function
- Binary cross-entropy (BCE) over the 128 pitch outputs
- Pos-weighting to handle sparsity of pianoroll
This model directly predicts dense 16×128 accompaniment for each bar.



## Key Challenges

1) Sparsity of pianoroll

Piano accompaniment contains few active notes per time step
Without proper weighting, models tend to output overly low probabilities or trivial solutions

Solve this by add loss to encourage the probability of some notes 

2) Prediction collapse

Some models collapse to repeating a single chord

Solve this by tuning of: context size, loss weighting, sparsity penalty, decoding thresholds


## Results
<img width="2560" height="1112" alt="image" src="https://github.com/user-attachments/assets/0d0dc20d-dff6-4ee2-8595-0a921dca00f6" />

Also, some sample generation can be see in the LSTM fold.


## Future Work

### Explore more expressive sequence models
Our current LSTM model captures local temporal patterns, but future work may investigate more powerful architectures (e.g., Transformer variants, diffusion models, or hierarchical RNNs) that can better model long-range harmonic structure and multi-voice interactions.

### Combine LSTM with probabilistic models
Because HMMs excel at capturing high-level harmonic transitions while LSTMs learn fine-grained temporal patterns, a hybrid system—e.g., HMM for chord-level structure + LSTM for frame-level realization—may leverage the strengths of both paradigms.

### Modular modeling for different musical roles
We are thinking about train separate models for chord progression, rhythmic patterns, and voice-leading, and then fuse the outputs. This may reduce complexity and improve controllability.

### Reverse-direction modeling: Piano → Melody
Since accompaniment often contains richer polyphonic information (multiple notes per timestep) while melody is typically monophonic, the reverse task—predicting melody from piano accompaniment—may be easier for sequence models and could provide useful insights into the melodic structure present in accompaniment.





