"""Simple chord HMM baseline skeleton.

This is a placeholder; real implementation will include:
- State space definition (chords / functional classes)
- Transition probabilities (estimated from corpus)
- Emission model (e.g., pitch-class set given chord)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

@dataclass
class HMMSpec:
    states: List[str]
    start_prob: np.ndarray  # shape (n_states,)
    trans_prob: np.ndarray  # shape (n_states, n_states)
    # Emission parameters placeholder

class ChordHMM:
    def __init__(self, spec: HMMSpec):
        self.spec = spec
        self.n_states = len(spec.states)

    def viterbi(self, emissions: Sequence[int]) -> List[str]:
        """Run Viterbi over a sequence of emission *indices* (placeholder).
        In a real version, emissions would be feature vectors -> likelihoods.
        """
        T = len(emissions)
        dp = np.full((self.n_states, T), -np.inf)
        back = np.zeros((self.n_states, T), dtype=int)

        dp[:, 0] = np.log(self.spec.start_prob + 1e-12)
        for t in range(1, T):
            for j in range(self.n_states):
                scores = dp[:, t-1] + np.log(self.spec.trans_prob[:, j] + 1e-12)
                back[j, t] = int(np.argmax(scores))
                dp[j, t] = np.max(scores)  # + log(emission_prob) placeholder

        # Backtrack
        last = int(np.argmax(dp[:, -1]))
        path_idx = [last]
        for t in range(T-1, 0, -1):
            last = back[last, t]
            path_idx.append(last)
        path_idx.reverse()
        return [self.spec.states[i] for i in path_idx]


def toy_spec(n_states: int = 4) -> HMMSpec:
    states = [f"C{i}" for i in range(n_states)]
    start = np.ones(n_states)
    start /= start.sum()
    trans = np.random.rand(n_states, n_states)
    trans /= trans.sum(axis=1, keepdims=True)
    return HMMSpec(states=states, start_prob=start, trans_prob=trans)

if __name__ == "__main__":
    spec = toy_spec()
    model = ChordHMM(spec)
    emissions = [0, 1, 2, 3, 0, 1]
    print(model.viterbi(emissions))
