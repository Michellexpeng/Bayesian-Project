"""
Key-Aware HDP-HSMM Model Definition
Path: src/models/hdp_hsmm.py
"""
import numpy as np

class KeyAwareHDPHSMM:
    """
    HDP-HSMM: Uses a frame-level approximation for efficient Gibbs sampling.
    """
    def __init__(self, n_max_states: int, obs_dim: int, prior_trans_matrix: np.ndarray = None):
        self.L = n_max_states
        self.V = obs_dim
        
        if prior_trans_matrix is not None:
            self.prior_bias = np.ones((self.L, self.L))
            k = min(prior_trans_matrix.shape[0], self.L)
            self.prior_bias[:k, :k] = prior_trans_matrix[:k, :k]
        else:
            self.prior_bias = np.ones((self.L, self.L))

        self.pi_star = np.ones(self.L) / self.L
        self.A = np.zeros((self.L, self.L))    
        self.dur_params = np.ones(self.L) * 5.0 
        self.B = np.ones((self.L, self.V)) / self.V 


        self.states = [] 

    def init_gibbs(self, sequences):

        self.states = []
        for seq in sequences:
            self.states.append(np.random.randint(0, self.L, size=len(seq)))

    def _get_frame_level_transition(self):

        A_frame = np.zeros((self.L, self.L))
        for i in range(self.L):

            avg_dur = max(1.1, self.dur_params[i])
            p_stay = 1.0 - (1.0 / avg_dur)
            p_leave = 1.0 / avg_dur
            
            A_frame[i, i] = p_stay

            for j in range(self.L):
                if i != j:
                    A_frame[i, j] = p_leave * self.A[i, j]
        
        return A_frame / (A_frame.sum(axis=1, keepdims=True) + 1e-10)

    def sample_parameters(self, sequences):
        
        transitions = np.zeros((self.L, self.L))
        dur_collections = [[] for _ in range(self.L)]
        emit_counts = np.zeros((self.L, self.V))

        for idx, seq_z in enumerate(self.states):
            seq_x = sequences[idx]
            
            np.add.at(emit_counts, (seq_z, seq_x), 1)
            
            if len(seq_z) == 0: continue
            
            curr_s = seq_z[0]
            curr_dur = 1
            
            for t in range(1, len(seq_z)):
                next_s = seq_z[t]
                if next_s == curr_s:
                    curr_dur += 1
                else:
                    dur_collections[curr_s].append(curr_dur)
                    transitions[curr_s, next_s] += 1
                    curr_s = next_s
                    curr_dur = 1
            dur_collections[curr_s].append(curr_dur)

        for i in range(self.L):
            self.A[i] = np.random.dirichlet(self.prior_bias[i] + transitions[i])

        for i in range(self.L):
            self.B[i] = np.random.dirichlet(1.0 + emit_counts[i])

        for i in range(self.L):
            durs = dur_collections[i]
            if durs:
                alpha_post = 1.0 + sum(durs)
                beta_post = 0.2 + len(durs)
                self.dur_params[i] = np.random.gamma(alpha_post, 1.0/beta_post)
            else:
                self.dur_params[i] = np.random.gamma(5.0, 1.0)

    def sample_states(self, sequences):

        A_frame = self._get_frame_level_transition()
        
        new_states = []
        for seq_x in sequences:
            T = len(seq_x)
            if T == 0:
                new_states.append(np.array([], dtype=np.int32))
                continue

            # --- Forward Pass (Filtering) ---
            log_alpha = np.zeros((T, self.L))
            
            log_alpha[0] = np.log(self.pi_star + 1e-10) + np.log(self.B[:, seq_x[0]] + 1e-10)
            
            for t in range(1, T):
                prev = log_alpha[t-1][:, np.newaxis] # (L, 1)
                trans = np.log(A_frame + 1e-10)      # (L, L)
                
                # Log-Sum-Exp 技巧
                scores = prev + trans 
                max_scores = np.max(scores, axis=0)
                log_sum_exp = max_scores + np.log(np.sum(np.exp(scores - max_scores), axis=0))
                
                log_alpha[t] = log_sum_exp + np.log(self.B[:, seq_x[t]] + 1e-10)
            
            # --- Backward Pass (Sampling) ---
            z_seq = np.zeros(T, dtype=np.int32)
            
            params = np.exp(log_alpha[T-1] - np.max(log_alpha[T-1]))
            params /= np.sum(params)
            z_seq[-1] = np.random.choice(self.L, p=params)
            
            for t in range(T-2, -1, -1):
                next_z = z_seq[t+1]
                # P(z_t | z_{t+1}, x) \propto alpha[t] * P(z_{t+1} | z_t)
                log_prob = log_alpha[t] + np.log(A_frame[:, next_z] + 1e-10)
                
                prob = np.exp(log_prob - np.max(log_prob))
                prob /= np.sum(prob)
                z_seq[t] = np.random.choice(self.L, p=prob)
                
            new_states.append(z_seq)
        
        self.states = new_states

    def generate(self, melody_length: int):
        """生成序列"""
        gen_obs = []
        curr_state = np.random.randint(0, self.L)
        t = 0
        while t < melody_length:

            dur = max(1, int(np.random.poisson(self.dur_params[curr_state])))

            obs_indices = np.random.choice(self.V, size=dur, p=self.B[curr_state])
            gen_obs.extend(obs_indices)

            curr_state = np.random.choice(self.L, p=self.A[curr_state])
            t += dur
            
        return gen_obs[:melody_length]