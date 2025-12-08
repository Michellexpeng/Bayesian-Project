import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import torch

# 引入你提供的处理脚本
# 使用显式的完整路径导入
from src.data import pop909_parser
from src.data import chord_preprocessing
from src.data import key_aware_features

class HarmonicDataset:
    """
    负责加载 POP909 数据并将其转换为 HDP-HSMM 所需的序列格式。
    集成 Key-Aware 特征提取。
    """
    def __init__(self, data_root: Path):
        self.data_root = data_root
        # 词汇表映射
        self.chord_vocab: Dict[str, int] = {}
        self.function_vocab: Dict[str, int] = {}
        self.inv_function_vocab: Dict[int, str] = {}
        
    def process_single_song(self, song_dir: Path) -> Dict[str, np.ndarray]:
        """
        处理单首歌曲：对齐 Beat -> 提取 Key-Aware 特征 -> 数字化
        """
        # 1. 加载原始标注 
        ann = pop909_parser.load_song(song_dir)
        if not ann.midi_path or not ann.key:
            return None

        # 2. 获取 Beat 时间网格 [cite: 3, 9]
        beat_times = [b.time for b in ann.beats]
        
        # 3. 将和弦对齐到 Beat 网格 [cite: 2]
        raw_chords = [(c.start, c.end, c.label) for c in ann.chords]
        aligned_chords = chord_preprocessing.align_chords_to_beats(raw_chords, beat_times)
        
        # 4. 提取 Key-Aware 特征 (Roman Numerals, Functions)
        # 使用 key_audio.txt 中的全局调性
        key_label = ann.key.label 
        features_list = key_aware_features.chord_sequence_to_features(aligned_chords, key_label)
        
        return {
            "song_id": ann.song_id,
            "key": key_label,
            "features": features_list,
            "beat_times": beat_times
        }

    def build_dataset(self, songs_data: List[Dict]):
        """构建数值化的观察序列"""
        all_features = []
        for s in songs_data:
            all_features.extend(s['features'])
            
        # 构建功能性词汇表 (例如: "IV:maj", "V:dom7")
        # 这将作为模型的"观察值"或者"先验指导"
        self.function_vocab = key_aware_features.build_functional_vocabulary(all_features)
        self.inv_function_vocab = {v: k for k, v in self.function_vocab.items()}
        
        sequences = []
        for s in songs_data:
            seq_indices = []
            for feat in s['features']:
                # 组合 Roman Numeral 和 Quality 作为唯一标识
                roman = feat['roman_numeral']
                quality = feat['chord_quality']
                label = f"{roman}:{quality}" if roman != 'N' else "N"
                
                idx = self.function_vocab.get(label, 0) # 0 for unknown/N
                seq_indices.append(idx)
            sequences.append(np.array(seq_indices, dtype=np.int32))
            
        return sequences

    def get_prior_matrix(self) -> np.ndarray:
        """
        构造 Key-Aware 转移先验矩阵 (Transition Prior Matrix).
        基于音乐理论 (Tonic -> Subdominant -> Dominant -> Tonic)
        """
        n_states = len(self.function_vocab)
        prior_matrix = np.ones((n_states, n_states)) * 1.0  # 默认为弱狄利克雷分布
        
        for i_label, i_idx in self.function_vocab.items():
            for j_label, j_idx in self.function_vocab.items():
                if i_label == "N" or j_label == "N":
                    continue
                
                # 解析功能 (Simple heuristic based on RNA)
                # 你可以根据 key_aware_features.get_harmonic_function 扩展这里的逻辑
                i_roman = i_label.split(":")[0]
                j_roman = j_label.split(":")[0]
                
                # 规则示例: V (Dominant) -> I (Tonic) 概率极高
                if "V" in i_roman and "I" in j_roman and "IV" not in j_roman:
                     prior_matrix[i_idx, j_idx] += 10.0 # 增强先验权重
                
                # 规则示例: IV (Subdom) -> V (Dominant)
                if "IV" in i_roman and "V" in j_roman:
                    prior_matrix[i_idx, j_idx] += 5.0
                    
        return prior_matrix

# 使用示例
if __name__ == "__main__":
    # 假设你的数据在当前目录的 'POP909' 文件夹下
    dataset = HarmonicDataset(Path("POP909"))
    # 加载前5首作为测试
    raw_data = [dataset.process_single_song(p) for p in pop909_parser.find_songs(Path("POP909"))[:5]]
    raw_data = [d for d in raw_data if d] # 过滤None
    
    obs_seqs = dataset.build_dataset(raw_data)
    prior = dataset.get_prior_matrix()
    print(f"Dataset created with {len(obs_seqs)} sequences.")
    print(f"Vocab size: {len(dataset.function_vocab)}")
    print(f"Sample Prior shape: {prior.shape}")