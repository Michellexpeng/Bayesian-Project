"""Compare HDP-HMM and HDP-HSMM models on the test set.

Strictly follows the 70/15/15 Data Split:
- Train:      0% - 70%  (Songs 0-635)
- Validation: 70% - 85% (Songs 636-771)
- Test:       85% - 100% (Songs 772-909)

Usage:
  python scripts/compare_hdp_models.py --hdp_hmm models/hdp_hmm.pkl --hdp_hsmm models/hdp_hsmm.pkl --pop909 data/POP909
"""
import argparse
import pickle
import sys
import numpy as np
from pathlib import Path
import time

# Add Project Root to Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hdp_dataset import HarmonicDataset
from src.data import pop909_parser

import scripts.test_hdp_hmm as test_hmm
import scripts.test_hdp_hsmm as test_hsmm

def load_model(path: Path):
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_test_sequences(data_root: Path, vocab):
    """Load and process TEST set sequences (Last 15%)."""
    print(f"Loading test data from {data_root}...")
    dataset = HarmonicDataset(data_root)
    
    # 1. 严格排序，保证顺序一致
    all_songs = sorted(pop909_parser.find_songs(data_root))
    N = len(all_songs)
    
    # 2. 计算切分点 (70% / 15% / 15%)
    idx_train_end = int(N * 0.70)
    idx_val_end = int(N * 0.85)
    
    # 3. 获取测试集 (最后 15%)
    test_songs = all_songs[idx_val_end:]
    
    # 打印切分详情，供用户核对
    print("\n" + "-" * 50)
    print("📊 DATASET SPLIT VERIFICATION")
    print("-" * 50)
    print(f"Total Songs:      {N}")
    print(f"Train (0-70%):    {idx_train_end} songs")
    print(f"Val   (70-85%):   {idx_val_end - idx_train_end} songs (Held-out)")
    print(f"Test  (85-100%):  {len(test_songs)} songs (Used for Evaluation)")
    print("-" * 50 + "\n")
    
    # 4. 处理数据
    raw_data = [dataset.process_single_song(p) for p in test_songs]
    raw_data = [d for d in raw_data if d]
    
    sequences = []
    for d in raw_data:
        seq = []
        for feat in d['features']:
            roman = feat['roman_numeral']
            qual = feat['chord_quality']
            lbl = f"{roman}:{qual}" if roman != 'N' else "N"
            seq.append(vocab.get(lbl, 0))
        sequences.append(np.array(seq, dtype=np.int32))
        
    return sequences, len(test_songs)

def main():
    parser = argparse.ArgumentParser(description="Compare HDP Models")
    parser.add_argument('--hdp_hmm', type=str, default='models/hdp_hmm.pkl')
    parser.add_argument('--hdp_hsmm', type=str, default='models/hdp_hsmm.pkl')
    parser.add_argument('--pop909', type=str, default='data/POP909')
    args = parser.parse_args()

    # --- 🔍 路径修正逻辑 (关键修改) ---
    project_root = Path(__file__).resolve().parents[1]

    # 1. 修正 HMM 模型路径
    hmm_path = Path(args.hdp_hmm)
    if not hmm_path.exists():
        hmm_path = project_root / args.hdp_hmm
    
    if not hmm_path.exists():
        print(f"❌ Error: HMM model not found at {hmm_path}")
        return

    # 2. 修正 HSMM 模型路径
    hsmm_path = Path(args.hdp_hsmm)
    if not hsmm_path.exists():
        hsmm_path = project_root / args.hdp_hsmm
        
    if not hsmm_path.exists():
        print(f"❌ Error: HSMM model not found at {hsmm_path}")
        return

    # 3. 修正数据路径
    data_path = Path(args.pop909)
    if not data_path.exists():
        data_path = project_root / args.pop909
        
    if not data_path.exists():
        print(f"❌ Error: Data folder not found at {data_path}")
        return
    # ------------------------------------

    print("\n" + "=" * 80)
    print("MODEL COMPARISON: HDP-HMM vs HDP-HSMM")
    print("=" * 80 + "\n")

    # 1. Load Models (使用修正后的路径)
    hmm_data = load_model(hmm_path)
    hsmm_data = load_model(hsmm_path)
    
    hmm_model = hmm_data['model']
    hsmm_model = hsmm_data['model']
    vocab = hmm_data['vocab'] 
    inv_vocab = {v: k for k, v in vocab.items()}

    # 2. Load Test Data (使用修正后的路径)
    test_seqs, n_test_songs = get_test_sequences(data_path, vocab)

    # 3. Evaluate HDP-HMM
    print(f"Evaluating HDP-HMM...")
    start_time = time.time()
    hmm_ppl, hmm_acc, _ = test_hmm.evaluate_hmm(
        test_seqs, hmm_model.pi, hmm_model.A, hmm_model.B, inv_vocab
    )
    print(f"  -> Done in {time.time() - start_time:.2f}s")

    # 4. Evaluate HDP-HSMM
    print(f"Evaluating HDP-HSMM...")
    start_time = time.time()
    pi_eq, A_eq = test_hsmm.convert_hsmm_to_hmm(
        hsmm_model.pi_star, hsmm_model.A, hsmm_model.dur_params
    )
    hsmm_ppl, hsmm_acc, _ = test_hmm.evaluate_hmm(
        test_seqs, pi_eq, A_eq, hsmm_model.B, inv_vocab
    )
    print(f"  -> Done in {time.time() - start_time:.2f}s")

    # 5. Comparison Table
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE COMPARISON (Test Set)")
    print("=" * 80)
    print(f"{'Metric':<20} | {'HDP-HMM':<15} | {'HDP-HSMM':<15} | {'Improvement':<15}")
    print("-" * 80)
    
    # Perplexity
    perp_imp = (hmm_ppl - hsmm_ppl) / hmm_ppl * 100
    better = "HSMM" if hsmm_ppl < hmm_ppl else "HMM"
    sign = "+" if hsmm_ppl < hmm_ppl else "-" 
    print(f"{'Perplexity':<20} | {hmm_ppl:<15.2f} | {hsmm_ppl:<15.2f} | {sign}{abs(perp_imp):.1f}% ({better})")
    
    # Accuracy
    acc_imp = (hsmm_acc - hmm_acc) / hmm_acc * 100
    better_acc = "HSMM" if hsmm_acc > hmm_acc else "HMM"
    sign_acc = "+" if hsmm_acc > hmm_acc else "-"
    print(f"{'Accuracy':<20} | {hmm_acc:<15.2%} | {hsmm_acc:<15.2%} | {sign_acc}{abs(acc_imp):.1f}% ({better_acc})")
    
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()