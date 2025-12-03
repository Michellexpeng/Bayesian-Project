"""Train Conditional HMM with separate transition matrices for major/minor modes.

This model learns:
- P(functional_chord | previous_functional_chord, key_mode)

Separate transition matrices for major vs minor reduces data sparsity.

Usage:
  python scripts/train_conditional.py --pop909 data/POP909 --out models/hmm_conditional.pkl
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pickle
import json
from datetime import datetime
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pop909_parser import load_dataset
from src.data.chord_preprocessing import align_chords_to_beats
from src.data.key_aware_features import extract_key_aware_features


def extract_functional_chord_sequence(song) -> tuple[list[str], str]:
    """Extract functional chord sequence with simplified labels."""
    if not song.key or not song.chords or not song.beats:
        return ([], "unknown")
    
    key_label = song.key.label
    beat_times = [b.time for b in song.beats if b.is_beat or b.is_downbeat]
    chord_tuples = [(c.start, c.end, c.label) for c in song.chords]
    
    chord_seq = align_chords_to_beats(chord_tuples, beat_times)
    
    functional_seq = []
    key_mode = None
    
    for chord_label in chord_seq:
        features = extract_key_aware_features(chord_label, key_label)
        
        if key_mode is None:
            key_mode = features['key_mode']
        
        # Simplified functional label: just roman numeral (no quality suffix)
        # This reduces vocabulary size dramatically
        roman = features['roman_numeral']
        
        if roman == 'N' or roman is None:
            func_label = 'N'
        else:
            func_label = roman  # e.g., "I", "IV", "V", "i", "iv", "v"
        
        functional_seq.append(func_label)
    
    return (functional_seq, key_mode or "major")


def split_dataset(songs: list, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    """Split songs into train/val/test sets."""
    np.random.seed(seed)
    indices = np.random.permutation(len(songs))
    
    n_train = int(len(songs) * train_ratio)
    n_val = int(len(songs) * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return (
        [songs[i] for i in train_idx],
        [songs[i] for i in val_idx],
        [songs[i] for i in test_idx],
    )


def estimate_conditional_hmm(
    sequences: list[list[int]], 
    key_modes: list[str],
    n_states: int
) -> dict:
    """Estimate separate HMM parameters for major and minor modes.
    
    Returns dict with:
      major_start_prob, major_trans_prob
      minor_start_prob, minor_trans_prob
    """
    # Separate sequences by mode
    major_seqs = [seq for seq, mode in zip(sequences, key_modes) if mode == 'major']
    minor_seqs = [seq for seq, mode in zip(sequences, key_modes) if mode == 'minor']
    
    def estimate_for_mode(seqs):
        start_count = np.zeros(n_states, dtype=float)
        trans_count = np.zeros((n_states, n_states), dtype=float)
        
        for seq in seqs:
            if len(seq) == 0:
                continue
            start_count[seq[0]] += 1.0
            for i in range(len(seq) - 1):
                trans_count[seq[i], seq[i + 1]] += 1.0
        
        # Add-one smoothing
        start_count += 1.0
        trans_count += 1.0
        
        start_prob = start_count / start_count.sum()
        trans_prob = trans_count / trans_count.sum(axis=1, keepdims=True)
        
        return start_prob, trans_prob
    
    major_start, major_trans = estimate_for_mode(major_seqs)
    minor_start, minor_trans = estimate_for_mode(minor_seqs)
    
    return {
        'major_start_prob': major_start,
        'major_trans_prob': major_trans,
        'minor_start_prob': minor_start,
        'minor_trans_prob': minor_trans,
        'n_major_songs': len(major_seqs),
        'n_minor_songs': len(minor_seqs),
    }


def compute_conditional_perplexity(
    sequences: list[list[int]], 
    key_modes: list[str],
    params: dict
) -> float:
    """Compute perplexity using mode-specific parameters."""
    total_log_prob = 0.0
    total_transitions = 0
    
    for seq, mode in zip(sequences, key_modes):
        if len(seq) == 0:
            continue
        
        # Select parameters based on mode
        if mode == 'major':
            start_prob = params['major_start_prob']
            trans_prob = params['major_trans_prob']
        else:
            start_prob = params['minor_start_prob']
            trans_prob = params['minor_trans_prob']
        
        total_log_prob += np.log(start_prob[seq[0]] + 1e-10)
        total_transitions += 1
        
        for i in range(len(seq) - 1):
            total_log_prob += np.log(trans_prob[seq[i], seq[i + 1]] + 1e-10)
            total_transitions += 1
    
    avg_log_prob = total_log_prob / total_transitions if total_transitions > 0 else 0
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909", type=str, required=True)
    ap.add_argument("--out", type=str, default="models/hmm_conditional.pkl")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    print("=" * 80)
    print(f"Conditional HMM Training (Mode-Specific) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load and split data
    print(f"\n[1/6] Loading POP909 ...")
    songs = load_dataset(Path(args.pop909))
    train_songs, val_songs, test_songs = split_dataset(songs, args.train_ratio, args.val_ratio, args.seed)
    print(f"  ✓ Train: {len(train_songs)}, Val: {len(val_songs)}, Test: {len(test_songs)}")
    
    # Extract functional sequences (simplified: roman only, no quality)
    print(f"\n[2/6] Extracting functional sequences (Roman numerals only) ...")
    
    def extract_all(song_list):
        seqs, modes = [], []
        for song in song_list:
            seq, mode = extract_functional_chord_sequence(song)
            if seq:
                seqs.append(seq)
                modes.append(mode)
        return seqs, modes
    
    train_seqs_str, train_modes = extract_all(train_songs)
    val_seqs_str, val_modes = extract_all(val_songs)
    test_seqs_str, test_modes = extract_all(test_songs)
    
    print(f"  ✓ Train: {len(train_seqs_str)} songs")
    print(f"  ✓ Val:   {len(val_seqs_str)} songs")
    print(f"  ✓ Test:  {len(test_seqs_str)} songs")
    
    # Build vocabulary (from all sequences to ensure consistency)
    all_chords = set()
    for seq in train_seqs_str:
        all_chords.update(seq)
    vocab = {ch: i for i, ch in enumerate(sorted(all_chords))}
    print(f"\n  📊 Vocabulary size: {len(vocab)} (simplified roman numerals)")
    
    # Show top chords
    all_train_chords = [ch for seq in train_seqs_str for ch in seq]
    chord_counts = Counter(all_train_chords)
    print(f"  📊 Top 15 functional chords: {chord_counts.most_common(15)}")
    
    # Count by mode
    major_count = sum(1 for m in train_modes if m == 'major')
    minor_count = sum(1 for m in train_modes if m == 'minor')
    print(f"  📊 Major: {major_count} songs ({major_count/len(train_modes)*100:.1f}%)")
    print(f"  📊 Minor: {minor_count} songs ({minor_count/len(train_modes)*100:.1f}%)")
    
    # Convert to integers
    def to_int_seqs(str_seqs, vocab):
        return [[vocab[ch] for ch in seq if ch in vocab] for seq in str_seqs]
    
    train_seqs = to_int_seqs(train_seqs_str, vocab)
    val_seqs = to_int_seqs(val_seqs_str, vocab)
    test_seqs = to_int_seqs(test_seqs_str, vocab)
    
    # Train conditional HMM
    print(f"\n[3/6] Training Conditional HMM (separate matrices for major/minor) ...")
    params = estimate_conditional_hmm(train_seqs, train_modes, len(vocab))
    print(f"  ✓ Major songs: {params['n_major_songs']}")
    print(f"  ✓ Minor songs: {params['n_minor_songs']}")
    
    # Compute perplexity
    print(f"\n[4/6] Computing perplexity ...")
    train_perp = compute_conditional_perplexity(train_seqs, train_modes, params)
    val_perp = compute_conditional_perplexity(val_seqs, val_modes, params)
    test_perp = compute_conditional_perplexity(test_seqs, test_modes, params)
    
    print(f"  ✓ Train perplexity: {train_perp:.2f}")
    print(f"  ✓ Val perplexity:   {val_perp:.2f}")
    print(f"  ✓ Test perplexity:  {test_perp:.2f}")
    
    # Analyze most common transitions by mode
    print(f"\n[5/6] Analyzing mode-specific patterns ...")
    
    def get_top_transitions(trans_prob, vocab_inv, top_k=5):
        flat_indices = np.argsort(trans_prob.flatten())[-top_k:][::-1]
        top_trans = []
        for idx in flat_indices:
            i, j = divmod(idx, trans_prob.shape[1])
            prob = trans_prob[i, j]
            top_trans.append((vocab_inv[i], vocab_inv[j], prob))
        return top_trans
    
    vocab_inv = {v: k for k, v in vocab.items()}
    
    major_top = get_top_transitions(params['major_trans_prob'], vocab_inv)
    minor_top = get_top_transitions(params['minor_trans_prob'], vocab_inv)
    
    print(f"  📊 Top 5 transitions in MAJOR:")
    for from_ch, to_ch, prob in major_top:
        print(f"      {from_ch} → {to_ch}: {prob:.3f}")
    
    print(f"  📊 Top 5 transitions in MINOR:")
    for from_ch, to_ch, prob in minor_top:
        print(f"      {from_ch} → {to_ch}: {prob:.3f}")
    
    # Save model
    print(f"\n[6/6] Saving model ...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    model = {
        "vocab": vocab,
        "inv_vocab": vocab_inv,
        "major_start_prob": params['major_start_prob'],
        "major_trans_prob": params['major_trans_prob'],
        "minor_start_prob": params['minor_start_prob'],
        "minor_trans_prob": params['minor_trans_prob'],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_type": "conditional_hmm",
            "n_train_songs": len(train_seqs),
            "n_val_songs": len(val_seqs),
            "n_test_songs": len(test_seqs),
            "vocab_size": len(vocab),
            "train_perplexity": float(train_perp),
            "val_perplexity": float(val_perp),
            "test_perplexity": float(test_perp),
            "n_major_train": params['n_major_songs'],
            "n_minor_train": params['n_minor_songs'],
        },
    }
    
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\n{'=' * 80}")
    print(f"✓ Model saved to {out_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
