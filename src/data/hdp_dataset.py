import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import torch


from src.data import pop909_parser
from src.data import chord_preprocessing
from src.data import key_aware_features

class HarmonicDataset:

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.chord_vocab: Dict[str, int] = {}
        self.function_vocab: Dict[str, int] = {}
        self.inv_function_vocab: Dict[int, str] = {}
        
    def process_single_song(self, song_dir: Path) -> Dict[str, np.ndarray]:

        ann = pop909_parser.load_song(song_dir)
        if not ann.midi_path or not ann.key:
            return None

        beat_times = [b.time for b in ann.beats]
        
        raw_chords = [(c.start, c.end, c.label) for c in ann.chords]
        aligned_chords = chord_preprocessing.align_chords_to_beats(raw_chords, beat_times)
        
        key_label = ann.key.label 
        features_list = key_aware_features.chord_sequence_to_features(aligned_chords, key_label)
        
        return {
            "song_id": ann.song_id,
            "key": key_label,
            "features": features_list,
            "beat_times": beat_times
        }

    def build_dataset(self, songs_data: List[Dict]):

        all_features = []
        for s in songs_data:
            all_features.extend(s['features'])
            
        self.function_vocab = key_aware_features.build_functional_vocabulary(all_features)
        self.inv_function_vocab = {v: k for k, v in self.function_vocab.items()}
        
        sequences = []
        for s in songs_data:
            seq_indices = []
            for feat in s['features']:

                roman = feat['roman_numeral']
                quality = feat['chord_quality']
                label = f"{roman}:{quality}" if roman != 'N' else "N"
                
                idx = self.function_vocab.get(label, 0) # 0 for unknown/N
                seq_indices.append(idx)
            sequences.append(np.array(seq_indices, dtype=np.int32))
            
        return sequences

    def get_prior_matrix(self) -> np.ndarray:

        n_states = len(self.function_vocab)
        prior_matrix = np.ones((n_states, n_states)) * 1.0  
        
        for i_label, i_idx in self.function_vocab.items():
            for j_label, j_idx in self.function_vocab.items():
                if i_label == "N" or j_label == "N":
                    continue
                
                i_roman = i_label.split(":")[0]
                j_roman = j_label.split(":")[0]
                
                if "V" in i_roman and "I" in j_roman and "IV" not in j_roman:
                     prior_matrix[i_idx, j_idx] += 10.0

                if "IV" in i_roman and "V" in j_roman:
                    prior_matrix[i_idx, j_idx] += 5.0
                    
        return prior_matrix

if __name__ == "__main__":

    dataset = HarmonicDataset(Path("POP909"))

    raw_data = [dataset.process_single_song(p) for p in pop909_parser.find_songs(Path("POP909"))[:5]]
    raw_data = [d for d in raw_data if d]

    obs_seqs = dataset.build_dataset(raw_data)
    prior = dataset.get_prior_matrix()
    print(f"Dataset created with {len(obs_seqs)} sequences.")
    print(f"Vocab size: {len(dataset.function_vocab)}")
    print(f"Sample Prior shape: {prior.shape}")