# Bayesian Conditional Harmonization (Symbolic Music)

Goal: Generate harmonization/accompaniment conditioned on a given melody, using POP909 and Classical Piano MIDI datasets. We will compare a simple HMM baseline, a Bayesian advanced model (HDP-HSMM/HDP-HMM with key-aware priors), and (optionally) a small LSTM/Transformer or API reference.

## Datasets
- POP909 (primary): melody–chord–piano aligned symbolic tracks for 909 pop songs.
  - Link: https://github.com/music-x-lab/POP909-Dataset
- Classical Piano MIDI (auxiliary): ~295 pieces by 19 classical composers (style sensitivity/generalization checks).
  - Link: https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi

## Method (high-level)
- Baseline: Markov/HMM over chord states with simple emissions/features.
- Bayesian advanced: HDP-HSMM/HDP-HMM with key-aware priors to capture variable-length harmonic segments and provide calibrated uncertainty; posterior sampling for diverse arrangements.
- Neural/API reference (optional): lightweight LSTM/Transformer trained from scratch, and/or an API/pretrained generator for context (not the primary focus).

## Evaluation
We will compare objective metrics (chord accuracy, functional consistency, voice-leading penalties, structure F1, likelihood/perplexity), and human listening tests.

## Repository layout
```
.
├── data/                 # place raw datasets here (git-ignored)
├── notebooks/            # optional exploration notebooks
├── scripts/              # runnable scripts (quickstart, baselines)
├── src/
│   ├── data/             # dataset loaders/preprocessing
│   ├── evaluation/       # metrics and evaluation utilities
│   ├── models/           # HMM baseline, Bayesian models, neural refs
│   └── utils/            # MIDI IO and music utilities
└── README.md
```

## Quickstart
1) Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Optional: point to a POP909 root with MIDI files and run a quick sanity check:

```bash
python scripts/quickstart.py --pop909 /path/to/POP909
```

This script will simply enumerate candidate songs and print a tiny toy HMM demo so you can verify the environment.

## Notes
- Heavy Bayesian libraries (NumPyro/Pyro) are not required for the quickstart and will be added when implementing the advanced model.
- For now, the baseline HMM uses only standard Python and NumPy (hmmlearn is optional later).