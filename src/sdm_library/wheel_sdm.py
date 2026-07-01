"""
sdm_library/wheel_sdm.py

(c) Wheel-model spiking SDM — thesis Chapters 6-7.

Same architecture as RankOrderSDM (significance vectors, MAX learning) but
address decode and data read use wheel-model spiking neuron dynamics.

The wheel model (thesis section 6.4, eq. 6.10-6.13):
  - Phase rises at constant slope m (the "spin rate")
  - On receiving spike with significance sigma, phase jumps by w_ij * sigma
  - Fires when phase crosses threshold Theta; resets to 0

Key property (why thesis uses this, not RDLIF): under high threshold
(Theta >> max cumulative jump), firing order equals rank by total weighted
input = exactly the dot-product ranking of the abstract model.
This gives EXACT equivalence between WheelSDM and RankOrderSDM outputs.

The N_a parameter here controls the address decoder weight sparsity
(a-of-A), same as in RankOrderSDM. Both N_i (input sparsity) and N_a
(decoder weight sparsity) are now correctly separate parameters.

Closed-form fire times used (no timestep simulation needed).
"""

import numpy as np
from .base import SDMBase, topk_indices


def wheel_fire_times(W_layer: np.ndarray, events: list,
                     slope: float = 1.0, threshold: float = 1000.0) -> np.ndarray:
    """Exact closed-form fire times for all neurons in a wheel-model layer.
    events: list of (source_index, significance) in arrival order."""
    num_neurons = W_layer.shape[0]
    cumulative = np.zeros(num_neurons)
    fired = np.zeros(num_neurons, dtype=bool)
    fire_t = np.full(num_neurons, np.inf)

    for k, (src_idx, sigma) in enumerate(events):
        t_k = float(k)
        if k > 0:
            prev_t = float(k - 1)
            T = (threshold - cumulative) / slope
            crosses = (~fired) & (T >= prev_t) & (T < t_k)
            fire_t[crosses] = T[crosses]
            fired |= crosses
        cumulative += W_layer[:, src_idx] * sigma
        over = (~fired) & (cumulative >= threshold)
        fire_t[over] = t_k
        fired |= over

    not_fired = ~fired
    if np.any(not_fired):
        T = (threshold - cumulative[not_fired]) / slope
        fire_t[not_fired] = float(len(events)) + T
    return fire_t


def sig_vec_to_events(vec: np.ndarray, N: int) -> list:
    """Convert significance vector to (index, value) event list,
    sorted by descending significance (earliest fire = highest weight)."""
    top = topk_indices(vec, N)
    top_sorted = top[np.argsort(vec[top])[::-1]]
    return [(int(idx), float(vec[idx])) for idx in top_sorted]


class WheelSDM(SDMBase):

    def __init__(self, D=256, N_i=11, N_a=20, W=4096, N_w=16, N_d=11,
                 alpha=0.99, seed=0, slope=1.0, threshold=1000.0):
        self.slope = slope
        self.threshold = threshold
        super().__init__(D, N_i, N_a, W, N_w, N_d, alpha, seed)

    def _init_weights(self):
        # Address decoder: W neurons, N_a significance-weighted connections
        self.W_addr = np.zeros((self.W, self.D), dtype=np.float64)
        for i in range(self.W):
            active = self.rng.choice(self.D, size=self.N_a, replace=False)
            order = self.rng.permutation(self.N_a)
            for rank, j in enumerate(order):
                self.W_addr[i, active[j]] = self.alpha ** rank
        self.W_data = np.zeros((self.D, self.W), dtype=np.float64)

    def _addr_forward(self, address_vec: np.ndarray) -> np.ndarray:
        events = sig_vec_to_events(address_vec, self.N_i)
        ft = wheel_fire_times(self.W_addr, events, self.slope, self.threshold)
        top = np.argsort(ft)[:self.N_w]
        top = top[np.isfinite(ft[top])]
        out = np.zeros(self.W, dtype=np.float64)
        for rank, idx in enumerate(top):
            out[idx] = self.alpha ** rank
        return out

    def _data_write(self, addr_decoded: np.ndarray, data_vec: np.ndarray):
        self.W_data = np.maximum(self.W_data,
                                  np.outer(data_vec, addr_decoded))

    def _data_read(self, addr_decoded: np.ndarray) -> np.ndarray:
        events = sig_vec_to_events(addr_decoded, self.N_w)
        ft = wheel_fire_times(self.W_data, events, self.slope, self.threshold)
        top = np.argsort(ft)[:self.N_d]
        top = top[np.isfinite(ft[top])]
        out = np.zeros(self.D, dtype=np.float64)
        for rank, idx in enumerate(top):
            out[idx] = self.alpha ** rank
        return out
