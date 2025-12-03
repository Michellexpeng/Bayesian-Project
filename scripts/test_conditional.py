"""Test Conditional HMM on test set with prediction accuracy.

Evaluates:
1. Next-chord prediction accuracy (给定前一个和弦，能否预测对下一个)
2. Perplexity (模型的不确定性)

Usage:
  python scripts/test_conditional.py --model models/hmm_conditional.pkl --pop909 data/POP909
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
        
        roman = features['roman_numeral']
        
        if roman == 'N' or roman is None:
            func_label = 'N'
        else:
            func_label = roman
        
        functional_seq.append(func_label)
    
    return (functional_seq, key_mode if key_mode else "unknown")


def compute_perplexity(sequences, modes, params):
    """Compute perplexity."""
    major_start = params['major_start_prob']
    major_trans = params['major_trans_prob']
    minor_start = params['minor_start_prob']
    minor_trans = params['minor_trans_prob']
    
    total_log_prob = 0.0
    total_tokens = 0
    
    for seq, mode in zip(sequences, modes):
        if len(seq) < 2:
            continue
        
        if mode == 'major':
            start_prob = major_start
            trans_prob = major_trans
        elif mode == 'minor':
            start_prob = minor_start
            trans_prob = minor_trans
        else:
            continue
        
        log_prob = np.log(start_prob[seq[0]] + 1e-10)
        
        for i in range(1, len(seq)):
            prev_chord = seq[i-1]
            curr_chord = seq[i]
            log_prob += np.log(trans_prob[prev_chord, curr_chord] + 1e-10)
        
        total_log_prob += log_prob
        total_tokens += len(seq)
    
    avg_log_prob = total_log_prob / total_tokens
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity


def evaluate_accuracy(sequences, modes, params, vocab_inv):
    """Evaluate next-chord prediction accuracy."""
    correct = 0
    total = 0
    
    # Track by mode
    mode_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Track confusion (most common errors)
    confusion = defaultdict(Counter)
    
    for seq, mode in zip(sequences, modes):
        if len(seq) < 2 or mode not in ['major', 'minor']:
            continue
        
        if mode == 'major':
            trans_prob = params['major_trans_prob']
        else:
            trans_prob = params['minor_trans_prob']
        
        # For each position (except first), predict next chord
        for i in range(1, len(seq)):
            prev_chord = seq[i-1]
            true_chord = seq[i]
            
            # Prediction: argmax of transition probability
            pred_chord = np.argmax(trans_prob[prev_chord])
            
            if pred_chord == true_chord:
                correct += 1
                mode_stats[mode]['correct'] += 1
            else:
                # Track error
                true_label = vocab_inv[true_chord]
                pred_label = vocab_inv[pred_chord]
                confusion[true_label][pred_label] += 1
            
            total += 1
            mode_stats[mode]['total'] += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'mode_stats': dict(mode_stats),
        'confusion': dict(confusion),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/hmm_conditional.pkl')
    parser.add_argument('--pop909', type=str, default='data/POP909')
    args = parser.parse_args()
    
    print(f"\n{'=' * 80}")
    print(f"TESTING CONDITIONAL HMM - PREDICTION ACCURACY ON TEST SET")
    print(f"{'=' * 80}\n")
    
    # Load model
    print(f"[1/4] Loading model from {args.model} ...")
    with open(args.model, "rb") as f:
        model = pickle.load(f)
    
    vocab = model['vocab']
    vocab_inv = model['inv_vocab']
    params = {
        'major_start_prob': model['major_start_prob'],
        'major_trans_prob': model['major_trans_prob'],
        'minor_start_prob': model['minor_start_prob'],
        'minor_trans_prob': model['minor_trans_prob'],
    }
    
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
    print(f"\n[3/4] Extracting functional chord sequences ...")
    test_seqs_str = []
    test_modes = []
    
    for song in test_songs:
        seq, mode = extract_functional_chord_sequence(song)
        if len(seq) >= 2 and mode in ['major', 'minor']:
            test_seqs_str.append(seq)
            test_modes.append(mode)
    
    print(f"  ✓ Valid test sequences: {len(test_seqs_str)}")
    
    # Convert to integers
    def to_int_seqs(str_seqs, vocab):
        return [[vocab[ch] for ch in seq if ch in vocab] for seq in str_seqs]
    
    test_seqs = to_int_seqs(test_seqs_str, vocab)
    
    # Evaluate
    print(f"\n[4/4] Evaluating predictions ...")
    
    # Perplexity
    test_perp = compute_perplexity(test_seqs, test_modes, params)
    print(f"  ✓ Test Perplexity: {test_perp:.2f}")
    
    # Accuracy
    results = evaluate_accuracy(test_seqs, test_modes, params, vocab_inv)
    accuracy = results['accuracy']
    correct = results['correct']
    total = results['total']
    
    print(f"  ✓ Test Accuracy: {accuracy:.2%} ({correct}/{total} correct predictions)")
    
    # By mode
    print(f"\n  📊 Accuracy by mode:")
    for mode, stats in results['mode_stats'].items():
        mode_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"      {mode.capitalize()}: {mode_acc:.2%} ({stats['correct']}/{stats['total']})")
    
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
