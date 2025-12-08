import sys
import os
import pickle
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.hdp_dataset import HarmonicDataset
from src.data import pop909_parser
from src.models.hdp_hmm import KeyAwareHDPHMM

def main():
    print("--- Training HDP-HMM (Standard) [Train Split Only] ---")
    
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "POP909"
    
    if not data_root.exists():
        print(f"❌ Error: Path does not exist: {data_root}")
        return

    dataset = HarmonicDataset(data_root)
    
    all_songs = sorted(pop909_parser.find_songs(data_root))
    n_total = len(all_songs)
    
    n_train = int(n_total * 0.70)
    
    train_songs = all_songs[:n_train]
    
    print(f"📊 Dataset Split:")
    print(f"   Total songs: {n_total}")
    print(f"   Training on: {len(train_songs)} songs (First 70%)")
    print(f"   (Validation/Test sets are held out)")

    print("🔄 Processing training data...")
    raw_data = [dataset.process_single_song(p) for p in train_songs]
    raw_data = [d for d in raw_data if d] 

    obs_seqs = dataset.build_dataset(raw_data)
    prior_matrix = dataset.get_prior_matrix()
    
    print("⚙️ Initializing HDP-HMM...")

    model = KeyAwareHDPHMM(
        n_max_states=24, 
        obs_dim=len(dataset.function_vocab), 
        prior_trans_matrix=prior_matrix
    )
    model.init_gibbs(obs_seqs)
    
    n_iter = 50
    print(f"🔥 Starting Gibbs Sampling for {n_iter} iterations...")
    for i in range(n_iter):
        if i % 10 == 0:
            print(f"   Iter {i}/{n_iter}")
        
        model.sample_parameters(obs_seqs)
        
        model.sample_states(obs_seqs) 
        
    save_path = project_root / "models" / "hdp_hmm.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_type": "HDP-HMM",
        "n_train_songs": len(train_songs),
        "split_ratio": 0.70,
        "iterations": n_iter
    }
    
    with open(save_path, "wb") as f:
        pickle.dump({
            "model": model, 
            "vocab": dataset.function_vocab,
            "metadata": metadata
        }, f)
        
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    main()