# Baseline HMM Model - Validation Report

**Model**: `models/hmm_baseline.pkl`  
**Trained**: 2025-12-03  
**Dataset**: POP909 (909 songs)

---

## ✅ Validation Results: PASSED

### 1. Data Split Overview

| Split | Songs | Beats | Unique Chords | Avg Beats/Song |
|-------|-------|-------|---------------|----------------|
| **Train** | 636 (70%) | 104,614 | 198 | 164.5 |
| **Val** | 136 (15%) | 22,399 | 165 | 164.7 |
| **Test** | 137 (15%) | 22,399 | 161 | 163.5 |
| **Total** | 909 | 149,412 | 198 | 164.4 |

---

## 2. Probability Distribution Validation ✓

**Status**: All probability distributions are valid

| Metric | Value | Status |
|--------|-------|--------|
| Start prob sum | 1.000000 | ✓ Valid |
| Start prob range | [0.0012, 0.5271] | ✓ Non-negative |
| Trans prob row sums | 1.000000 ± 0.000000 | ✓ Valid |
| Trans prob range | [0.00006, 0.6074] | ✓ Non-negative |

**Interpretation**: The HMM parameters are mathematically sound. All probabilities sum to 1 and are non-negative.

---

## 3. Chord Distribution Analysis

**Vocabulary Size**: 198 unique chords (after transposition to C/Am)

### Top 10 Most Frequent Chords

| Rank | Chord | Count | Percentage |
|------|-------|-------|------------|
| 1 | A:min | 16,812 | 16.0% |
| 2 | G:maj | 14,811 | 14.1% |
| 3 | C:maj | 14,284 | 13.6% |
| 4 | F:maj | 11,444 | 10.9% |
| 5 | D:min | 7,577 | 7.2% |
| 6 | E:min | 6,536 | 6.2% |
| 7 | E:maj | 2,676 | 2.5% |
| 8 | N (no chord) | 1,769 | 1.7% |
| 9 | A:maj | 1,744 | 1.7% |
| 10 | A:min7 | 1,571 | 1.5% |

### Coverage Analysis

- **Top 10 chords**: 75.4% of all beats
- **Top 20 chords**: 85.8% of all beats

**Interpretation**: The dataset is dominated by basic triads in C major (C, F, G) and A minor (A, D, E), which is expected for pop music. The long tail (193 total chords) includes extended harmonies and inversions.

---

## 4. Model Performance Analysis ✓

**Status**: No overfitting detected, good generalization

### Perplexity

| Split | Perplexity | Assessment |
|-------|------------|------------|
| **Train** | 11.61 | Baseline |
| **Val** | 12.20 | +5.1% vs train |
| **Test** | 11.44 | -1.5% vs train |

**Overfit Ratio**: 1.051 (val/train)  
**Threshold**: < 1.5 for acceptable generalization

**Interpretation**:
- ✓ Low perplexity indicates the model captures training data patterns well
- ✓ Val and test perplexities are very close to train (< 10% gap)
- ✓ No significant overfitting (ratio 1.05 << 1.5 threshold)
- ✓ Model generalizes well to unseen data

**What does perplexity mean?**
- Perplexity = exp(cross-entropy) = geometric mean of branching factor
- Perplexity of 11.44 means the model is "uncertain between ~11 chords" on average
- Lower is better (perfect model would have perplexity = 1)
- For comparison: random uniform model over 193 chords would have perplexity = 193

### Prediction Accuracy

| Split | Accuracy | Correct Predictions | Total Transitions |
|-------|----------|---------------------|-------------------|
| **Test** | 33.99% | 7,613 | 22,399 |

**Interpretation**:
- Model correctly predicts 1 out of every 3 chord transitions
- Performance limited by large vocabulary (193 chords) and polyphonic complexity
- Top error: F:maj → G:maj (1,093 incorrect predictions)
- Confusion mainly between closely related chords in the same key

---

## 5. Transition Matrix Analysis

**Sparsity**: 0.0% (all transitions have non-zero probability due to add-one smoothing)

### Self-Transition Statistics
- **Average**: 0.0733 (7.3% chance of staying on same chord)
- **Range**: [0.059, 0.607]

### Top 5 Chord Transitions

| From Chord | To Chord | Probability | Musical Context |
|------------|----------|-------------|-----------------|
| E:maj | A:min | 0.3712 | V → i (dominant to tonic in A minor) |
| E:min/5 | A:min | 0.3043 | v/5 → i (voice leading to tonic) |
| F:maj | G:maj | 0.2871 | IV → V (subdominant to dominant) |
| D:min | G:maj | 0.2841 | ii → V (pre-dominant to dominant) |
| C:dom7 | F:maj | 0.2765 | V7 → IV (secondary dominant) |

**Interpretation**: The model captures musically meaningful transitions:
- Dominant → tonic resolutions (E → Am)
- Functional progressions (IV → V, ii → V)
- Secondary dominants (C7 → F)

---

## 6. Test Set Performance ✓

**Next-Chord Prediction Accuracy**: 33.99% (7,613 / 22,399 predictions)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 33.99% |
| **Correct Predictions** | 7,613 |
| **Total Predictions** | 22,399 |
| **Wrong Predictions** | 14,786 |
| **Test Perplexity** | 11.44 |

### Performance Interpretation

**Prediction Accuracy: 33.99%**
- **Random baseline**: 0.52% (1/193 chords)
- **Our model**: 33.99% ≈ **65× better than random**
- Model has learned real chord transition patterns, but still has significant uncertainty
---

## 8. Music Theory Consistency ✓
| Rank | True Chord | Predicted | Count | Analysis |
|------|------------|-----------|-------|----------|
| 1 | F:maj | G:maj | 1,093 | IV → V vs IV → other progressions |
| 2 | C:maj | G:maj | 1,006 | I → V most common, but I can go many ways |
| 3 | E:min | A:min | 757 | iii → vi (both tonic function in C major) |
| 4 | F:maj | A:min | 721 | IV → vi (deceptive resolution confusion) |
| 5 | D:min | A:min | 608 | ii → vi vs ii → V confusion |
| 6 | G:maj | D:min | 473 | V → ii (retrogression, unusual) |
| 7 | A:min | G:maj | 455 | vi → V vs vi → other |
| 8 | E:min | G:maj | 369 | iii → V (less common progression) |
| 9 | G:maj | A:min | 357 | V → vi (deceptive cadence) |
| 10 | A:min | C:maj | 327 | vi → I (relative minor to major) |

**Key Issue**: Errors show the model confuses **absolute chords across different keys**. For example:
- `F:maj → G:maj` is predicted because IV→V is common in C major
- But the true continuation might be different if we're in a different key
- The model doesn't understand functional harmony (I, IV, V) - only absolute pitches

---

## 7. Data Coverage Analysis ✓

**Vocabulary Coverage on Unseen Data**:

| Split | Unique Chords | Coverage | Status |
|-------|---------------|----------|--------|
| Val | 159 / 193 | 82.4% | ✓ Good |
| Test | 153 / 193 | 79.3% | ✓ Good |

**Out-of-Vocabulary (OOV) Chords**:
- Val: 34 chords not seen in training (17.6%)
- Test: 40 chords not seen in training (20.7%)

**Interpretation**: 
- ✓ >75% coverage indicates good vocabulary generalization
- OOV chords are likely rare extended harmonies (e.g., augmented 6ths, Neapolitan chords)
- Add-one smoothing ensures non-zero probability for unseen transitions

---

## 7. Music Theory Consistency ✓

### Common Functional Progressions

| Progression | Probability | Musical Function |
|-------------|-------------|------------------|
| G:maj → C:maj | 0.2529 | V → I (authentic cadence) |
| F:maj → C:maj | 0.0935 | IV → I (plagal cadence) |
| C:maj → G:maj | 0.0778 | I → V (half cadence setup) |
| C:maj → F:maj | 0.0729 | I → IV (subdominant) |
| A:min → E:maj | 0.0081 | i → V (minor key dominant) |

**Dominant → Tonic Resolution**:
- Average probability: 0.1442 (14.4%)
- This is significantly higher than random (1/193 = 0.52%)

**Interpretation**:
- ✓ Model learned functional harmony patterns
- ✓ V → I progressions have high probability (25.3% for G → C)
- ✓ Cadential patterns are well-represented
- ⚠ Minor key dominant (A:min → E:maj) is weaker (0.81%), suggesting pop music preference for modal harmony

---

## 9. Visualizations

Generated visualizations in `validation_results/`:

1. **`chord_distribution.png`**: Bar chart of top 20 chords
2. **`perplexity_comparison.png`**: Perplexity across train/val/test splits
3. **`transition_matrix.png`**: Heatmap of transition probabilities (top 30 chords)

---

## Overall Assessment

### ✅ Strengths

1. **Valid Probability Distributions**: All mathematical constraints satisfied
2. **No Overfitting**: Excellent generalization (overfit ratio 1.05)
3. **Good Coverage**: 80%+ vocabulary coverage on val/test sets
4. **Music Theory Consistency**: Captures functional harmony progressions
5. **Reasonable Perplexity**: 11.44 on test set (65× better than random)
6. **Decent Prediction Accuracy**: 33.99% next-chord prediction accuracy
### ⚠️ Limitations

1. **Moderate Perplexity**: Perplexity of 11.44 means uncertain between ~11 chords on average
2. **Limited Prediction Accuracy**: 33.99% means wrong 2 out of 3 times
3. **No Functional Harmony**: Uses absolute chords (C, F, G) instead of Roman numerals (I, IV, V)
   - Treats C:maj in different keys as the same chord
   - Doesn't generalize across keys
4. **No Melody Conditioning**: Model ignores melody, only looks at previous chord
5. **1st-order Markov**: No long-range harmonic context (e.g., doesn't know if in verse/chorus)
6. **Large Vocabulary**: 193 chords leads to data sparsity
4. **Modal Harmony Weakness**: Pop songs often use modal interchange, which is hard for strict functional models

### 🎯 Next Steps for Improvement

1. **Add Melody Features**: Condition transitions on melody pitch, interval, contour
2. **Increase Context Window**: Use 2nd or 3rd order Markov (bigram/trigram chords)
### Expected Performance Gains

| Model | Test Perplexity | Test Accuracy | Improvement | Notes |
|-------|-----------------|---------------|-------------|-------|
| **Current Baseline** | 11.44 | 33.99% | - | Absolute chords, 1st-order |
| + Key-aware priors | ~5-6 | ~40-42% | +18-24% | **✅ ACHIEVED** (see Conditional HMM) |
| + 2nd-order Markov | ~4-5 | ~45-50% | +32-47% | Longer harmonic context |
| + Melody features | ~3-4 | ~50-55% | +47-62% | Pitch-aware transitions |
| **Full Bayesian Model** | ~3-4 | ~55-60% | +62-76% | All features combined |

**✅ UPDATE**: The Mode-Conditional HMM has achieved the "Key-aware priors" improvement:
- Test Perplexity: **4.96** (56.6% reduction from 11.44)
## Conclusion

The HMM baseline model is **mathematically valid** and **generalizes well** to unseen data. It successfully captures basic chord transition patterns and shows no signs of overfitting.

**Current Performance**:
- Test Perplexity: **11.44** (uncertain between ~11 chords)
- Test Accuracy: **33.99%** (correct 1 out of 3 predictions)
- 65× better than random guessing

**Main Limitation**: Uses absolute chords (C:maj, F:maj) instead of functional harmony (I, IV, V), which:
- Doesn't generalize across keys
- Leads to data sparsity (193 chords vs ~20 functional chords)
- Confuses functionally similar chords in different keys

**✅ Next Step COMPLETED**: Mode-Conditional HMM
- ✅ Implemented functional harmony (Roman numerals)
- ✅ Mode-conditional modeling (separate major/minor matrices)
- ✅ **Results**: Perplexity 4.96 (↓56.6%), Accuracy 41.82% (↑23%)
- See `CONDITIONAL_SUMMARY.md` for full results

**Future Improvements**:
1. **2nd-order Markov**: Condition on previous 2 chords → Expected perplexity ~4-5, accuracy ~45-50%
2. **Melody conditioning**: Use melody pitch/interval → Expected perplexity ~3-4, accuracy ~50-55%
3. **Hierarchical structure**: Variable-length segments → More musical phrasinging.

However, the high perplexity (11.61) and expected low chord accuracy (~5-10%) confirm that this is a **performance floor** rather than a production model. The model's main limitation is the lack of melody conditioning and short context window (0-order Markov).

The next phase should implement:
1. **Melody feature extraction** from MIDI tracks
2. **Higher-order n-gram models** (bigram/trigram chords)
3. **Bayesian HDP-HSMM** with key-aware priors for variable-length harmonic segments

These improvements should reduce perplexity to ~4-5 and increase chord accuracy to 40-50%, making the model suitable for creative applications like automatic harmonization and chord suggestion.

---

**Validation completed**: 2025-12-02  
**Status**: ✅ PASSED - Ready for next model iteration
