"""
Key-Aware HDP-HMM Model Definition (Standard HMM)
Path: src/models/hdp_hmm.py
"""
import numpy as np

class KeyAwareHDPHMM:
    """HDP-HMM: Standard Infinite HMM with Key-Aware Priors."""
    
    def __init__(self, n_max_states: int, obs_dim: int, prior_trans_matrix: np.ndarray = None):
        self.L = n_max_states
        self.V = obs_dim
        
        # Key-Aware Bias
        if prior_trans_matrix is not None:
            self.prior_bias = np.ones((self.L, self.L))
            k = min(prior_trans_matrix.shape[0], self.L)
            self.prior_bias[:k, :k] = prior_trans_matrix[:k, :k]
        else:
            self.prior_bias = np.ones((self.L, self.L))

        self.pi = np.ones(self.L) / self.L
        self.A = np.zeros((self.L, self.L))
        self.B = np.ones((self.L, self.V)) / self.V
        self.states = []

    def init_gibbs(self, sequences):
        self.states = []
        for seq in sequences:
            self.states.append(np.random.randint(0, self.L, size=len(seq)))

    def sample_parameters(self, sequences):
        # 1. Transitions
        trans_counts = np.zeros((self.L, self.L))
        for seq_z in self.states:
            for t in range(len(seq_z) - 1):
                trans_counts[seq_z[t], seq_z[t+1]] += 1
        
        for i in range(self.L):
            self.A[i] = np.random.dirichlet(self.prior_bias[i] + trans_counts[i])

        # 2. Emissions
        emit_counts = np.zeros((self.L, self.V))
        for idx, seq_x in enumerate(sequences):
            seq_z = self.states[idx]
            np.add.at(emit_counts, (seq_z, seq_x), 1)
        
        for i in range(self.L):
            self.B[i] = np.random.dirichlet(1.0 + emit_counts[i])

    def sample_states(self, sequences):
        """Standard Forward-Filtering Backward-Sampling (FFBS)."""
        new_states = []
        for seq_x in sequences:
            T = len(seq_x)
            if T == 0:
                new_states.append(np.array([], dtype=np.int32))
                continue
                
            # Forward
            log_alpha = np.zeros((T, self.L))
            log_alpha[0] = np.log(self.pi + 1e-10) + np.log(self.B[:, seq_x[0]] + 1e-10)
            
            for t in range(1, T):
                prev = log_alpha[t-1][:, np.newaxis]
                trans = np.log(self.A + 1e-10)
                scores = prev + trans 
                max_scores = np.max(scores, axis=0)
                log_sum = max_scores + np.log(np.sum(np.exp(scores - max_scores), axis=0))
                log_alpha[t] = log_sum + np.log(self.B[:, seq_x[t]] + 1e-10)
            
            # Backward
            z_seq = np.zeros(T, dtype=np.int32)
            params = np.exp(log_alpha[T-1] - np.max(log_alpha[T-1]))
            params /= np.sum(params)
            z_seq[-1] = np.random.choice(self.L, p=params)
            
            for t in range(T-2, -1, -1):
                next_z = z_seq[t+1]
                log_prob = log_alpha[t] + np.log(self.A[:, next_z] + 1e-10)
                prob = np.exp(log_prob - np.max(log_prob))
                prob /= np.sum(prob)
                z_seq[t] = np.random.choice(self.L, p=prob)
            new_states.append(z_seq)
        self.states = new_states

    def generate(self, length: int):
        gen_obs = []
        curr_state = np.random.randint(0, self.L)
        for _ in range(length):
            obs = np.random.choice(self.V, p=self.B[curr_state])
            gen_obs.append(obs)
            curr_state = np.random.choice(self.L, p=self.A[curr_state])
        return gen_obs