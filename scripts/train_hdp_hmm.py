import sys
import os
import pickle
import numpy as np
from pathlib import Path

# 添加项目根目录
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.hdp_dataset import HarmonicDataset
from src.data import pop909_parser
from src.models.hdp_hmm import KeyAwareHDPHMM

def main():
    print("--- Training HDP-HMM (Standard) [Train Split Only] ---")
    
    # 1. 设置路径
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "POP909"
    
    if not data_root.exists():
        print(f"❌ Error: Path does not exist: {data_root}")
        return

    dataset = HarmonicDataset(data_root)
    
    # 2. 加载并切分数据
    # 必须排序以确保每次切分一致
    all_songs = sorted(pop909_parser.find_songs(data_root))
    n_total = len(all_songs)
    
    # 按照 70% / 15% / 15% 切分
    n_train = int(n_total * 0.70)
    
    # 仅使用训练集！
    train_songs = all_songs[:n_train]
    
    print(f"📊 Dataset Split:")
    print(f"   Total songs: {n_total}")
    print(f"   Training on: {len(train_songs)} songs (First 70%)")
    print(f"   (Validation/Test sets are held out)")

    # 3. 处理数据
    print("🔄 Processing training data...")
    raw_data = [dataset.process_single_song(p) for p in train_songs]
    raw_data = [d for d in raw_data if d] # 过滤无效数据
    
    obs_seqs = dataset.build_dataset(raw_data)
    prior_matrix = dataset.get_prior_matrix()
    
    # 4. 初始化模型
    print("⚙️ Initializing HDP-HMM...")
    # 使用较大的状态上限，让 HDP 自动收缩
    model = KeyAwareHDPHMM(
        n_max_states=24, 
        obs_dim=len(dataset.function_vocab), 
        prior_trans_matrix=prior_matrix
    )
    model.init_gibbs(obs_seqs)
    
    # 5. 训练循环
    n_iter = 50
    print(f"🔥 Starting Gibbs Sampling for {n_iter} iterations...")
    for i in range(n_iter):
        if i % 10 == 0:
            print(f"   Iter {i}/{n_iter}")
        
        model.sample_parameters(obs_seqs)
        
        # 关键！必须取消注释！
        model.sample_states(obs_seqs) 
        
    # 6. 保存
    save_path = project_root / "models" / "hdp_hmm.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存元数据以便对比
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