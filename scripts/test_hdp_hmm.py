"""Test HDP-HMM on test set with perplexity and prediction accuracy.

Usage:
  python scripts/test_hdp_hmm.py --model models/hdp_hmm.pkl --pop909 data/POP909
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pickle
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hdp_dataset import HarmonicDataset
from src.data import pop909_parser

def hmm_forward(obs_seq, pi, A, B):
    """
    Standard HMM Forward Algorithm (in log domain).
    Returns:
        log_likelihood: P(X | model)
        alphas: Forward variables (log probs), shape (T, N_states)
    """
    T = len(obs_seq)
    N = A.shape[0]
    
    # alphas[t, i] = log P(x_1...x_t, z_t=i)
    alphas = np.zeros((T, N))
    
    # 1. Initialization
    # alpha_0(i) = log(pi_i) + log(P(x_0 | z_0=i))
    with np.errstate(divide='ignore'):
        alphas[0] = np.log(pi + 1e-10) + np.log(B[:, obs_seq[0]] + 1e-10)
    
    # 2. Recursion
    for t in range(1, T):
        for j in range(N):
            # alpha_t(j) = log_sum_exp(alpha_{t-1}(i) + log(A_ij)) + log(B_j(x_t))
            # Log-sum-exp trick for numerical stability
            prev_log_probs = alphas[t-1] + np.log(A[:, j] + 1e-10)
            max_log = np.max(prev_log_probs)
            log_sum = max_log + np.log(np.sum(np.exp(prev_log_probs - max_log)))
            
            alphas[t, j] = log_sum + np.log(B[j, obs_seq[t]] + 1e-10)
            
    # 3. Termination
    # log P(X) = log_sum_exp(alpha_T(i))
    last_log_probs = alphas[-1]
    max_log = np.max(last_log_probs)
    log_likelihood = max_log + np.log(np.sum(np.exp(last_log_probs - max_log)))
    
    return log_likelihood, alphas

def evaluate_hmm(sequences, pi, A, B, inv_vocab):
    """Evaluate Perplexity and Accuracy for HMM."""
    total_log_prob = 0.0
    total_tokens = 0
    correct = 0
    total_preds = 0
    
    confusion = defaultdict(Counter)
    
    print(f"Evaluating {len(sequences)} sequences...")
    
    for seq in sequences:
        if len(seq) < 2: continue
        
        # --- 1. Perplexity (Log Likelihood) ---
        log_prob, alphas = hmm_forward(seq, pi, A, B)
        total_log_prob += log_prob
        total_tokens += len(seq)
        
        # --- 2. Accuracy (Next Step Prediction) ---
        # Predict x_{t+1} given x_{1...t}
        # P(x_{t+1} | x_{1...t}) = \sum_j P(x_{t+1}|z_{t+1}=j) * \sum_i P(z_{t+1}=j|z_t=i) * P(z_t=i|x_{1...t})
        # P(z_t=i|x_{1...t}) is proportional to exp(alpha_t[i])
        
        N = A.shape[0]
        V = B.shape[1]
        
        for t in range(len(seq) - 1):
            # Belief state at t: P(z_t | x_{1...t})
            # Softmax over alphas[t]
            log_belief = alphas[t] - np.max(alphas[t]) # normalize for stability
            belief = np.exp(log_belief)
            belief /= np.sum(belief) # normalize to sum to 1
            
            # Predict next observation distribution
            # next_obs_dist = Belief @ A @ B
            next_state_dist = belief @ A # Shape (N,)
            next_obs_dist = next_state_dist @ B # Shape (V,)
            
            # Prediction: argmax
            pred_token = np.argmax(next_obs_dist)
            true_token = seq[t+1]
            
            if pred_token == true_token:
                correct += 1
            else:
                true_lbl = inv_vocab.get(true_token, "?")
                pred_lbl = inv_vocab.get(pred_token, "?")
                confusion[true_lbl][pred_lbl] += 1
                
            total_preds += 1

    # Metrics
    perplexity = np.exp(-total_log_prob / total_tokens)
    accuracy = correct / total_preds if total_preds > 0 else 0
    
    return perplexity, accuracy, confusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/hdp_hmm.pkl')
    parser.add_argument('--pop909', type=str, default='data/POP909')
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = project_root / args.model
        
    if not model_path.exists():
        print(f"❌ Error: Model file not found at: {model_path}")
        print(f"   Current working directory: {Path.cwd()}")
        return

    data_root = Path(args.pop909)
    if not data_root.exists():
        data_root = project_root / args.pop909
        
    if not data_root.exists():
        print(f"❌ Error: Data folder not found at: {data_root}")
        return
    
    print(f"\n{'=' * 80}")
    print(f"TESTING HDP-HMM - PREDICTION ACCURACY")
    print(f"{'=' * 80}\n")
    
    # 1. Load Model
    print(f"[1/4] Loading model from {model_path} ...")
    with open(model_path, "rb") as f: 
        data = pickle.load(f)
    
    model = data["model"]
    vocab = data["vocab"]
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # Extract HMM parameters
    pi = model.pi
    A = model.A
    B = model.B
    
    print(f"  ✓ States: {len(pi)}, Vocab: {len(vocab)}")
    
    # 2. Load Data (Last 15%)
    print(f"\n[2/4] Loading POP909 test set ...")
    
    dataset = HarmonicDataset(data_root)

    all_songs = sorted(pop909_parser.find_songs(data_root))
    
    # Use exact same split logic as training (First 70% Train, Next 15% Val, Last 15% Test)
    N = len(all_songs)
    idx_val_end = int(N * 0.85)
    
    test_songs_paths = all_songs[idx_val_end:]
    print(f"  ✓ Test set: {len(test_songs_paths)} songs (Index {idx_val_end}-{N})")
    
    # 3. Process Data
    print(f"\n[3/4] extracting feature sequences ...")
    raw_data = [dataset.process_single_song(p) for p in test_songs_paths]
    raw_data = [d for d in raw_data if d]
    
    # Ensure we use the model's vocab!
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
        
    print(f"  ✓ Valid sequences: {len(test_seqs)}")
    
    # 4. Evaluate
    print(f"\n[4/4] Evaluating (Forward Algorithm) ...")
    ppl, acc, confusion = evaluate_hmm(test_seqs, pi, A, B, inv_vocab)
    
    print(f"\n{'=' * 80}")
    print(f"HDP-HMM RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"Perplexity:        {ppl:.2f}")
    print(f"Prediction Acc:    {acc:.2%}")
    print(f"{'=' * 80}")
    
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