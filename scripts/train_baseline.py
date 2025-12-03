"""Train a simple HMM baseline on POP909 chord sequences.

Usage:
  python scripts/train_baseline.py --pop909 data/POP909 --out models/hmm_baseline.pkl
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pop909_parser import load_dataset
from src.data.chord_preprocessing import (
    normalize_chord,
    build_vocabulary,
    align_chords_to_beats,
    parse_chord_label,
    ROOT_TO_SEMITONE,
)


def compute_transpose_semitones(key_label: str | None) -> int:
    """Compute semitones to transpose to C major or A minor."""
    if key_label is None:
        return 0
    root, mode = parse_chord_label(key_label)
    if root is None:
        return 0
    root_semi = ROOT_TO_SEMITONE.get(root, 0)
    # Transpose to C (major) or A (minor)
    if "min" in mode.lower():
        target = 9  # A
    else:
        target = 0  # C
    return (target - root_semi) % 12


def extract_beat_aligned_chords(song, transpose: bool = True) -> list[str]:
    """Extract beat-aligned chord sequence, optionally transposed to C/Am."""
    beat_times = [b.time for b in song.beats if b.is_beat or b.is_downbeat]
    chord_tuples = [(c.start, c.end, c.label) for c in song.chords]
    raw_seq = align_chords_to_beats(chord_tuples, beat_times)
    
    if transpose and song.key:
        shift = compute_transpose_semitones(song.key.label)
        raw_seq = [normalize_chord(ch, shift) for ch in raw_seq]
    else:
        raw_seq = [normalize_chord(ch, 0) for ch in raw_seq]
    
    return raw_seq


def estimate_hmm(sequences: list[list[int]], n_states: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate start_prob and trans_prob from chord sequences (integer tokens).
    
    Returns:
      start_prob: shape (n_states,)
      trans_prob: shape (n_states, n_states)
    """
    start_count = np.zeros(n_states, dtype=float)
    trans_count = np.zeros((n_states, n_states), dtype=float)
    
    for seq in sequences:
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


def compute_perplexity(sequences, start_prob, trans_prob):
    """Compute perplexity on a set of sequences."""
    total_log_prob = 0.0
    total_transitions = 0
    
    for seq in sequences:
        if len(seq) == 0:
            continue
        
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
    ap.add_argument("--pop909", type=str, required=True, help="Path to POP909 root")
    ap.add_argument("--out", type=str, default="models/hmm_baseline.pkl", help="Output model file")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of songs (for quick tests)")
    ap.add_argument("--no-transpose", action="store_true", help="Disable transposition to C/Am")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    from datetime import datetime
    from collections import Counter
    
    print("=" * 80)
    print(f"Baseline HMM Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load and split data
    print(f"\n[1/6] Loading POP909 ...")
    all_songs = load_dataset(Path(args.pop909), limit=args.limit)
    
    # Use random split to match conditional model
    np.random.seed(args.seed)
    indices = np.random.permutation(len(all_songs))
    
    n_train = int(len(all_songs) * args.train_ratio)
    n_val = int(len(all_songs) * args.val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_songs = [all_songs[i] for i in train_idx]
    val_songs = [all_songs[i] for i in val_idx]
    test_songs = [all_songs[i] for i in test_idx]
    
    print(f"  ✓ Train: {len(train_songs)}, Val: {len(val_songs)}, Test: {len(test_songs)}")
    
    # Extract beat-aligned chord sequences
    print(f"\n[2/6] Extracting beat-aligned chord sequences (transposed to C/Am) ...")
    
    def extract_all(song_list):
        seqs = []
        for song in song_list:
            seq = extract_beat_aligned_chords(song, transpose=not args.no_transpose)
            if seq:
                seqs.append(seq)
        return seqs
    
    train_seqs_str = extract_all(train_songs)
    val_seqs_str = extract_all(val_songs)
    test_seqs_str = extract_all(test_songs)
    
    print(f"  ✓ Train: {len(train_seqs_str)} songs")
    print(f"  ✓ Val:   {len(val_seqs_str)} songs")
    print(f"  ✓ Test:  {len(test_seqs_str)} songs")
    
    # Build vocabulary from training set
    all_chords = []
    for seq in train_seqs_str:
        all_chords.extend(seq)
    
    vocab = build_vocabulary(all_chords)
    print(f"\n  📊 Vocabulary size: {len(vocab)} (transposed chords)")
    
    # Show top chords
    chord_counts = Counter(all_chords)
    print(f"  📊 Top 15 chords: {chord_counts.most_common(15)}")
    
    # Convert to integers
    def to_int_seqs(str_seqs, vocab):
        return [[vocab[ch] for ch in seq if ch in vocab] for seq in str_seqs]
    
    train_seqs = to_int_seqs(train_seqs_str, vocab)
    val_seqs = to_int_seqs(val_seqs_str, vocab)
    test_seqs = to_int_seqs(test_seqs_str, vocab)
    
    # Train HMM
    print(f"\n[3/6] Training Baseline HMM ...")
    start_prob, trans_prob = estimate_hmm(train_seqs, len(vocab))
    print(f"  ✓ Parameters estimated")
    
    # Compute perplexity
    print(f"\n[4/6] Computing perplexity ...")
    train_perp = compute_perplexity(train_seqs, start_prob, trans_prob)
    val_perp = compute_perplexity(val_seqs, start_prob, trans_prob)
    test_perp = compute_perplexity(test_seqs, start_prob, trans_prob)
    
    print(f"  ✓ Train perplexity: {train_perp:.2f}")
    print(f"  ✓ Val perplexity:   {val_perp:.2f}")
    print(f"  ✓ Test perplexity:  {test_perp:.2f}")
    
    # Analyze most common transitions
    print(f"\n[5/6] Analyzing transition patterns ...")
    
    def get_top_transitions(trans_prob, vocab_inv, top_k=5):
        flat_indices = np.argsort(trans_prob.flatten())[-top_k:][::-1]
        top_trans = []
        for idx in flat_indices:
            i, j = divmod(idx, trans_prob.shape[1])
            prob = trans_prob[i, j]
            top_trans.append((vocab_inv[i], vocab_inv[j], prob))
        return top_trans
    
    vocab_inv = {v: k for k, v in vocab.items()}
    top_trans = get_top_transitions(trans_prob, vocab_inv)
    
    print(f"  📊 Top 5 transitions:")
    for from_ch, to_ch, prob in top_trans:
        print(f"      {from_ch} → {to_ch}: {prob:.3f}")
    
    # Save model
    print(f"\n[6/6] Saving model ...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    model = {
        "vocab": vocab,
        "inv_vocab": vocab_inv,
        "start_prob": start_prob,
        "trans_prob": trans_prob,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_type": "baseline_hmm",
            "n_train_songs": len(train_seqs),
            "n_val_songs": len(val_seqs),
            "n_test_songs": len(test_seqs),
            "vocab_size": len(vocab),
            "train_perplexity": float(train_perp),
            "val_perplexity": float(val_perp),
            "test_perplexity": float(test_perp),
            "trained_date": datetime.now().strftime("%Y-%m-%d"),
        }
    }
    
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\n{'=' * 80}")
    print(f"✓ Model saved to {out_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
