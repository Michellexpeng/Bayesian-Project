"""Compare Baseline and Conditional HMM models.

Usage:
  python scripts/compare_models.py --baseline models/hmm_baseline.pkl --conditional models/hmm_conditional.pkl
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path


def load_model(path: Path) -> dict:
    """Load a trained model."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default='models/hmm_baseline.pkl')
    parser.add_argument('--conditional', type=str, default='models/hmm_conditional.pkl')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: Baseline HMM vs Conditional HMM")
    print("=" * 80 + "\n")
    
    # Load models
    baseline = load_model(Path(args.baseline))
    conditional = load_model(Path(args.conditional))
    
    baseline_meta = baseline.get('metadata', {})
    conditional_meta = conditional.get('metadata', {})
    
    # Extract metrics
    baseline_vocab = baseline_meta.get('vocab_size', len(baseline.get('vocab', {})))
    conditional_vocab = conditional_meta.get('vocab_size', len(conditional.get('vocab', {})))
    
    baseline_train_perp = baseline_meta.get('train_perplexity', 'N/A')
    baseline_val_perp = baseline_meta.get('val_perplexity', 'N/A')
    baseline_test_perp = baseline_meta.get('test_perplexity', 'N/A')
    
    conditional_train_perp = conditional_meta.get('train_perplexity', 'N/A')
    conditional_val_perp = conditional_meta.get('val_perplexity', 'N/A')
    conditional_test_perp = conditional_meta.get('test_perplexity', 'N/A')
    
    # Get accuracy from external test results (if available)
    baseline_acc = baseline_meta.get('test_accuracy', 'N/A')
    conditional_acc = conditional_meta.get('test_accuracy', 'N/A')
    
    # Print comparison table
    print("📊 PERFORMANCE COMPARISON:")
    print("="  * 80)
    print(f"{'Metric':<25} | {'Baseline HMM':<15} | {'Conditional HMM':<15} | {'Improvement':<12}")
    print("-" * 80)
    
    # Vocabulary size
    if baseline_vocab != 'N/A' and conditional_vocab != 'N/A':
        vocab_improve = (baseline_vocab - conditional_vocab) / baseline_vocab * 100
        print(f"{'Vocabulary Size':<25} | {baseline_vocab:<15} | {conditional_vocab:<15} | {vocab_improve:+.1f}%")
    
    # Train perplexity
    if baseline_train_perp != 'N/A' and conditional_train_perp != 'N/A':
        train_improve = (baseline_train_perp - conditional_train_perp) / baseline_train_perp * 100
        print(f"{'Train Perplexity':<25} | {baseline_train_perp:<15.2f} | {conditional_train_perp:<15.2f} | {train_improve:+.1f}%")
    
    # Val perplexity
    if baseline_val_perp != 'N/A' and conditional_val_perp != 'N/A':
        val_improve = (baseline_val_perp - conditional_val_perp) / baseline_val_perp * 100
        print(f"{'Val Perplexity':<25} | {baseline_val_perp:<15.2f} | {conditional_val_perp:<15.2f} | {val_improve:+.1f}%")
    
    # Test perplexity
    if baseline_test_perp != 'N/A' and conditional_test_perp != 'N/A':
        test_improve = (baseline_test_perp - conditional_test_perp) / baseline_test_perp * 100
        print(f"{'Test Perplexity':<25} | {baseline_test_perp:<15.2f} | {conditional_test_perp:<15.2f} | {test_improve:+.1f}%")
    
    # Test accuracy
    if baseline_acc != 'N/A' and conditional_acc != 'N/A':
        acc_improve = (conditional_acc - baseline_acc) / baseline_acc * 100
        print(f"{'Test Accuracy (%)':<25} | {baseline_acc:<15.2f} | {conditional_acc:<15.2f} | {acc_improve:+.1f}%")
    
    print("="  * 80)
    
    # Model details
    print("\n📋 MODEL DETAILS:")
    print("=" * 80)
    print(f"\nBaseline HMM:")
    print(f"  - Model type: {baseline_meta.get('model_type', 'baseline_hmm')}")
    print(f"  - Trained: {baseline_meta.get('trained_date', baseline_meta.get('timestamp', 'N/A'))}")
    print(f"  - Train songs: {baseline_meta.get('n_train_songs', 'N/A')}")
    print(f"  - Val songs: {baseline_meta.get('n_val_songs', 'N/A')}")
    print(f"  - Test songs: {baseline_meta.get('n_test_songs', 'N/A')}")
    
    print(f"\nConditional HMM:")
    print(f"  - Model type: {conditional_meta.get('model_type', 'conditional_hmm')}")
    print(f"  - Trained: {conditional_meta.get('trained_date', conditional_meta.get('timestamp', 'N/A'))}")
    print(f"  - Train songs: {conditional_meta.get('n_train_songs', 'N/A')}")
    print(f"  - Val songs: {conditional_meta.get('n_val_songs', 'N/A')}")
    print(f"  - Test songs: {conditional_meta.get('n_test_songs', 'N/A')}")
    print(f"  - Major songs: {conditional_meta.get('n_major_train', 'N/A')}")
    print(f"  - Minor songs: {conditional_meta.get('n_minor_train', 'N/A')}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
