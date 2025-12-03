# HMM Model Training Guide

## 📋 Overview

This project implements two approaches for chord sequence modeling:
- **Baseline HMM**: Standard HMM with transposition-based chord representation
- **Conditional HMM**: Mode-conditional HMM with functional harmony

## 🚀 Quick Start

### 1. Train Models

#### Train Baseline HMM
```bash
python scripts/train_baseline.py \
    --pop909 data/POP909 \
    --out models/hmm_baseline.pkl \
    --seed 42
```

#### Train Conditional HMM
```bash
python scripts/train_conditional.py \
    --pop909 data/POP909 \
    --out models/hmm_conditional.pkl \
    --seed 42
```

**Important**: Both models must use the **same seed** to ensure consistent data splits!

### 2. Compare Models

```bash
python scripts/compare_models.py \
    --baseline models/hmm_baseline.pkl \
    --conditional models/hmm_conditional.pkl
```

### 3. Test Models

#### Test Baseline HMM
```bash
python scripts/test_baseline.py \
    --model models/hmm_baseline.pkl \
    --pop909 data/POP909
```

#### Test Conditional HMM
```bash
python scripts/test_conditional.py \
    --model models/hmm_conditional.pkl \
    --pop909 data/POP909
```

## 📊 Latest Results (2025-12-03)

Using identical random splits (seed=42):

| Metric | Baseline HMM | Conditional HMM | Improvement |
|--------|--------------|-----------------|-------------|
| **Vocabulary Size** | 193 | 20 | **↓️89.6%** |
| **Train Perplexity** | 11.61 | 5.04 | **↓️56.6%** |
| **Val Perplexity** | 12.20 | 5.02 | **↓️58.9%** |
| **Test Perplexity** | 11.44 | 4.96 | **↓️56.6%** |
| **Test Accuracy** | 33.99% | 41.82% | **↑️23.0%** |

## 🔧 Parameters

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pop909` | Required | Path to POP909 dataset |
| `--out` | `models/hmm_*.pkl` | Output model file |
| `--seed` | 42 | Random seed (for reproducibility) |
| `--train-ratio` | 0.7 | Training set ratio |
| `--val-ratio` | 0.15 | Validation set ratio |
| `--limit` | None | Limit number of songs (for debugging) |

### Baseline-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--no-transpose` | False | Disable transposition to C/Am |

## 📁 Output Files

### Model File Structure

```python
{
    "vocab": dict,           # Chord → integer mapping
    "inv_vocab": dict,       # Integer → chord mapping
    "start_prob": np.array,  # Initial state probabilities
    "trans_prob": np.array,  # Transition probability matrix
    "metadata": {
        "model_type": str,
        "timestamp": str,
        "vocab_size": int,
        "n_train_songs": int,
        "n_val_songs": int,
        "n_test_songs": int,
        "train_perplexity": float,
        "val_perplexity": float,
        "test_perplexity": float,
        "test_accuracy": float,  # Test set accuracy (%)
        # Conditional-specific:
        "n_major_train": int,
        "n_minor_train": int
    }
}
```

### Conditional HMM Additional Fields

```python
{
    "major_start_prob": np.array,
    "major_trans_prob": np.array,
    "minor_start_prob": np.array,
    "minor_trans_prob": np.array
}
```

## 🔍 Training Workflow Comparison

### Baseline HMM

```
1. Load data → Random split (seed=42)
2. Extract chord sequences → Transpose to C major/A minor
3. Build vocabulary (193 chords)
4. Train single HMM
5. Calculate perplexity
6. Analyze transition patterns
```

### Conditional HMM

```
1. Load data → Random split (seed=42)
2. Extract functional chords → Roman numeral notation
3. Build vocabulary (20 functional chords)
4. Separate by mode → Train major/minor independent HMMs
5. Calculate conditional perplexity
6. Analyze mode-specific patterns
```

## 📈 Key Improvements

### Fixed Issues

✅ **Unified data splits**: Both models now use identical random split strategy  
✅ **Consistent output format**: Unified training log format  
✅ **Removed hardcoded values**: Conditional model no longer contains hardcoded baseline results  
✅ **Independent comparison script**: Dedicated script for model comparison  

### Code Quality Enhancements

- 🎯 Modular design
- 📊 Detailed training logs
- 🔄 Reproducible experiments (fixed seed)
- 📦 Complete metadata preservation
- 🧪 Independent test and comparison scripts

## 🎓 Model Details

### Baseline HMM

**Characteristics**:
- Transpose all songs to C major or A minor
- Single unified transition matrix
- Vocabulary size: 193 chords (including various chord qualities)

**Advantages**:
- Simple and intuitive
- Computationally efficient

**Disadvantages**:
- Large vocabulary, sparse data
- Cannot distinguish major/minor mode patterns

### Conditional HMM

**Characteristics**:
- Extract functional chords (Roman numerals)
- Separate major/minor transition matrices
- Vocabulary size: 20 functional chords

**Advantages**:
- Small vocabulary, sufficient data
- Captures mode-specific harmonic patterns
- Significantly lower perplexity (56.6% reduction)
- Significantly higher accuracy (23.0% improvement)

**Disadvantages**:
- Requires key identification
- Slightly more complex (two matrices)

## 🔬 Evaluation Metrics

### Perplexity

$$\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i)\right)$$

- **Lower is better**
- Measures model uncertainty
- Baseline: 11.44 → Conditional: 4.96 (56.6% improvement)

### Prediction Accuracy

- Given previous chord, predict next chord
- Uses greedy strategy: `argmax P(next|prev)`
- Baseline: 33.99% → Conditional: 41.82% (23.0% improvement)
- Calculated by `test_*.py` scripts, saved in model metadata

## 🛠️ Troubleshooting

### Common Issues

**Q: Models produce inconsistent results?**
A: Ensure both models use the same `--seed` parameter

**Q: Out of memory?**
A: Use `--limit 100` to restrict number of training songs

**Q: Cannot find model file?**
A: Check if `models/` directory exists

## 📚 Citation

If you use this code, please cite:

```bibtex
@misc{pop909_hmm,
  title={Hierarchical Bayesian Models for Chord Progression Analysis},
  author={Your Name},
  year={2025}
}
```

## 📞 Contact

For questions or suggestions, please refer to:
- `validation_results/BASELINE_SUMMARY.md` - Detailed baseline model analysis
- `validation_results/CONDITIONAL_SUMMARY.md` - Detailed conditional model analysis
- `README.md` - Project overview
