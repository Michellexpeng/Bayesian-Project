"""Test Baseline HMM on test set with prediction accuracy.

Usage:
  python scripts/test_baseline.py --model models/hmm_baseline.pkl --pop909 data/POP909
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pickle
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pop909_parser import load_dataset
from scripts.train_baseline import extract_beat_aligned_chords


def compute_perplexity(sequences, start_prob, trans_prob):
    """Compute perplexity."""
    total_log_prob = 0.0
    total_tokens = 0
    
    for seq in sequences:
        if len(seq) < 2:
            continue
        
        # First token
        log_prob = np.log(start_prob[seq[0]] + 1e-10)
        
        # Subsequent tokens
        for i in range(1, len(seq)):
            prev_chord = seq[i-1]
            curr_chord = seq[i]
            log_prob += np.log(trans_prob[prev_chord, curr_chord] + 1e-10)
        
        total_log_prob += log_prob
        total_tokens += len(seq)
    
    avg_log_prob = total_log_prob / total_tokens
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity


def evaluate_accuracy(sequences, trans_prob, vocab_inv):
    """Evaluate next-chord prediction accuracy."""
    correct = 0
    total = 0
    confusion = defaultdict(Counter)
    
    for seq in sequences:
        if len(seq) < 2:
            continue
        
        # For each position (except first), predict next chord
        for i in range(1, len(seq)):
            prev_chord = seq[i-1]
            true_chord = seq[i]
            
            # Prediction: argmax of transition probability
            pred_chord = np.argmax(trans_prob[prev_chord])
            
            if pred_chord == true_chord:
                correct += 1
            else:
                # Track error
                true_label = vocab_inv[true_chord]
                pred_label = vocab_inv[pred_chord]
                confusion[true_label][pred_label] += 1
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'confusion': dict(confusion),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/hmm_baseline.pkl')
    parser.add_argument('--pop909', type=str, default='data/POP909')
    args = parser.parse_args()
    
    print(f"\n{'=' * 80}")
    print(f"TESTING BASELINE HMM - PREDICTION ACCURACY ON TEST SET")
    print(f"{'=' * 80}\n")
    
    # Load model
    print(f"[1/4] Loading model from {args.model} ...")
    with open(args.model, "rb") as f:
        model = pickle.load(f)
    
    vocab = model['vocab']
    vocab_inv = model['inv_vocab']
    start_prob = model['start_prob']
    trans_prob = model['trans_prob']
    
    print(f"  ✓ Vocabulary size: {len(vocab)}")
    if 'metadata' in model:
        print(f"  ✓ Trained on {model['metadata'].get('n_train_songs', '?')} songs")
    
    # Load dataset
    print(f"\n[2/4] Loading POP909 test set ...")
    pop909_path = Path(args.pop909)
    songs = load_dataset(pop909_path)
    
    # Test set: last 15% (songs 774-909)
    test_songs = songs[773:]
    print(f"  ✓ Test set: {len(test_songs)} songs (songs 774-909)")
    
    # Extract sequences
    print(f"\n[3/4] Extracting chord sequences (transposed to C/Am) ...")
    test_seqs_str = []
    
    for song in test_songs:
        seq = extract_beat_aligned_chords(song, transpose=True)
        if len(seq) >= 2:
            test_seqs_str.append(seq)
    
    print(f"  ✓ Valid test sequences: {len(test_seqs_str)}")
    
    # Convert to integers
    def to_int_seqs(str_seqs, vocab):
        return [[vocab.get(ch, 0) for ch in seq] for seq in str_seqs]
    
    test_seqs = to_int_seqs(test_seqs_str, vocab)
    
    # Evaluate
    print(f"\n[4/4] Evaluating predictions ...")
    
    # Perplexity
    test_perp = compute_perplexity(test_seqs, start_prob, trans_prob)
    print(f"  ✓ Test Perplexity: {test_perp:.2f}")
    
    # Accuracy
    results = evaluate_accuracy(test_seqs, trans_prob, vocab_inv)
    accuracy = results['accuracy']
    correct = results['correct']
    total = results['total']
    
    print(f"  ✓ Test Accuracy: {accuracy:.2%} ({correct}/{total} correct predictions)")
    
    # Top errors
    print(f"\n  📊 Top 10 prediction errors:")
    all_errors = []
    for true_ch, pred_counts in results['confusion'].items():
        for pred_ch, count in pred_counts.items():
            all_errors.append((true_ch, pred_ch, count))
    
    all_errors.sort(key=lambda x: x[2], reverse=True)
    for true_ch, pred_ch, count in all_errors[:10]:
        print(f"      True: {true_ch} → Predicted: {pred_ch} ({count} times)")
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"TEST RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"Perplexity:            {test_perp:.2f}  (lower is better)")
    print(f"Prediction Accuracy:   {accuracy:.2%}  (higher is better)")
    print(f"Total Predictions:     {total}")
    print(f"Correct Predictions:   {correct}")
    print(f"Wrong Predictions:     {total - correct}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
