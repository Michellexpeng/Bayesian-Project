"""Test HDP-HSMM on test set.

Approximation:
  Converts HSMM parameters (Transitions + Duration Lambda) into an 
  equivalent frame-level HMM to compute Perplexity and Prediction Accuracy.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pickle
from collections import defaultdict, Counter

# Reuse logic from test_hdp_hmm
import test_hdp_hmm # We will reuse evaluate_hmm function
from src.data.hdp_dataset import HarmonicDataset
from src.data import pop909_parser

# Add Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def convert_hsmm_to_hmm(pi_star, A_hsmm, dur_params):
    """
    Convert HSMM parameters to an explicit frame-level HMM.
    
    HSMM State i:
      - Expected duration E[d] = dur_params[i] (Lambda for Poisson)
      - Transition to j: A_hsmm[i, j]
      
    Equivalent HMM State i:
      - Self-transition: a_ii = 1 - 1/E[d]
      - Transition to j: a_ij = (1/E[d]) * A_hsmm[i, j]
    """
    N = len(pi_star)
    A_hmm = np.zeros((N, N))
    
    for i in range(N):
        # Average duration (Poisson lambda. Note: Lambda usually means E[d])
        # We ensure min duration is slightly > 1 to allow self-transition
        avg_dur = max(1.1, dur_params[i]) 
        
        prob_leave = 1.0 / avg_dur
        prob_stay = 1.0 - prob_leave
        
        # Self-transition
        A_hmm[i, i] = prob_stay
        
        # Transitions to others
        # Distribute the 'prob_leave' mass according to A_hsmm
        for j in range(N):
            if i == j: continue # A_hsmm usually has 0 on diagonal, but just in case
            A_hmm[i, j] = prob_leave * A_hsmm[i, j]
            
    # Normalize rows to be sure
    row_sums = A_hmm.sum(axis=1, keepdims=True)
    A_hmm = A_hmm / (row_sums + 1e-10)
    
    return pi_star, A_hmm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/hdp_hsmm.pkl')
    parser.add_argument('--pop909', type=str, default='data/POP909')
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = project_root / args.model
    if not model_path.exists():
        print(f"❌ Error: Model file not found at: {model_path}")
        return

    data_root = Path(args.pop909)
    if not data_root.exists():
        data_root = project_root / args.pop909
    if not data_root.exists():
        print(f"❌ Error: Data folder not found at: {data_root}")
        return
    # -----------------------
    
    print(f"\n{'=' * 80}")
    print(f"TESTING HDP-HSMM - VIA HMM APPROXIMATION")
    print(f"{'=' * 80}\n")
    
    # 1. Load Model
    print(f"[1/4] Loading model from {model_path} ...")
    with open(model_path, "rb") as f: # 使用 model_path
        data = pickle.load(f)
        
    model = data["model"]
    vocab = data["vocab"]
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # 2. Convert HSMM -> HMM for evaluation
    print(f"[2/4] Converting HSMM parameters to equivalent HMM ...")
    pi_star = model.pi_star
    A_hsmm = model.A
    B = model.B
    dur_params = model.dur_params
    
    pi_hmm, A_hmm = convert_hsmm_to_hmm(pi_star, A_hsmm, dur_params)
    
    # 3. Load Data
    print(f"[3/4] Loading POP909 test set ...")
    # 使用 data_root
    dataset = HarmonicDataset(data_root)
    all_songs = sorted(pop909_parser.find_songs(data_root))
    
    N = len(all_songs)
    idx_val_end = int(N * 0.85)
    test_songs_paths = all_songs[idx_val_end:]
    print(f"  ✓ Test set: {len(test_songs_paths)} songs")
    
    # Process Data
    raw_data = [dataset.process_single_song(p) for p in test_songs_paths]
    raw_data = [d for d in raw_data if d]
    
    dataset.function_vocab = vocab
    
    test_seqs = []
    for d in raw_data:
        seq = []
        for feat in d['features']:
            roman = feat['roman_numeral']
            qual = feat['chord_quality']
            lbl = f"{roman}:{qual}" if roman != 'N' else "N"
            seq.append(vocab.get(lbl, 0))
        test_seqs.append(np.array(seq, dtype=np.int32))
        
    # 4. Evaluate using the imported HMM evaluator
    print(f"\n[4/4] Evaluating (Forward Algorithm on Equivalent HMM) ...")
    ppl, acc, confusion = test_hdp_hmm.evaluate_hmm(test_seqs, pi_hmm, A_hmm, B, inv_vocab)
    
    print(f"\n{'=' * 80}")
    print(f"HDP-HSMM RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"Perplexity:        {ppl:.2f}")
    print(f"Prediction Acc:    {acc:.2%}")
    print(f"{'=' * 80}")
    print("Note: Metrics are based on a frame-level HMM approximation")
    print("      of the HSMM's duration parameters.")
    
    # Top Errors
    print(f"\n  📊 Top 5 Errors:")
    errs = []
    for t, preds in confusion.items():
        for p, c in preds.items():
            errs.append((t, p, c))
    errs.sort(key=lambda x: x[2], reverse=True)
    for t, p, c in errs[:5]:
        print(f"     True: {t: <10} -> Pred: {p: <10} ({c} times)")

if __name__ == "__main__":
    main()
