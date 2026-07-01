"""
sdm_library/rdlif_sdm.py

(d) RDLIF spiking SDM — thesis Chapter 6, WIRN/IJCNN/ICANN 2005 papers.

Same architecture as RankOrderSDM but decode uses RDLIF spiking dynamics.

RDLIF (thesis section 6.2, WIRN paper eq. 1-2):
  dr_i/dt = sum_j w_ij * x_j(t) - (r_i - r0) / tau_r
  da_i/dt = r_i - (a_i - a0) / tau_a
  fires when a_i >= Theta

Key limitation (thesis p.125): RDLIF fire order depends on BOTH weighted
inputs AND absolute inter-spike timing intervals. This prevents exact
equivalence with the abstract model under interference — documented as
the reason the thesis switches to the wheel model for the full machine.

N_a controls address decoder weight sparsity (a-of-A), same as other
variants. N_i and N_a are correctly separate parameters here.
"""

import numpy as np
from .base import SDMBase, topk_indices
from .wheel_sdm import sig_vec_to_events


def rdlif_fire_times(W_layer: np.ndarray, events: list,
                     tau_r=0.5, tau_a=2.0, threshold=0.05,
                     r0=0.0, a0=0.0, t_window=15.0, dt=0.01) -> np.ndarray:
    """Euler integration of RDLIF ODE, return per-neuron fire times."""
    num_neurons = W_layer.shape[0]
    r = np.full(num_neurons, r0, dtype=np.float64)
    a = np.full(num_neurons, a0, dtype=np.float64)
    fired = np.zeros(num_neurons, dtype=bool)
    fire_t = np.full(num_neurons, np.inf)

    event_at = {}
    for k, (src_idx, sigma) in enumerate(events):
        event_at.setdefault(float(k), []).append((src_idx, sigma))

    num_steps = int(t_window / dt)
    for step in range(num_steps):
        t = step * dt
        t_key = round(t)
        if t_key in event_at and abs(t - t_key) < dt / 2:
            for src_idx, sigma in event_at[t_key]:
                r += W_layer[:, src_idx] * sigma
            del event_at[t_key]
        r += (-(r - r0) / tau_r) * dt
        a += (r - (a - a0) / tau_a) * dt
        crosses = (~fired) & (a >= threshold)
        fire_t[crosses] = t
        fired |= crosses
        a[fired] = a0
        r[fired] = r0
        if np.all(fired):
            break
    return fire_t


class RDLIFSDM(SDMBase):

    def __init__(self, D=256, N_i=11, N_a=20, W=4096, N_w=16, N_d=11,
                 alpha=0.99, seed=0,
                 tau_r=0.5, tau_a=2.0, threshold=0.05, t_window=15.0):
        self.tau_r = tau_r
        self.tau_a = tau_a
        self.threshold = threshold
        self.t_window = t_window
        super().__init__(D, N_i, N_a, W, N_w, N_d, alpha, seed)

    def _init_weights(self):
        self.W_addr = np.zeros((self.W, self.D), dtype=np.float64)
        for i in range(self.W):
            active = self.rng.choice(self.D, size=self.N_a, replace=False)
            order = self.rng.permutation(self.N_a)
            for rank, j in enumerate(order):
                self.W_addr[i, active[j]] = self.alpha ** rank
        self.W_data = np.zeros((self.D, self.W), dtype=np.float64)

    def _addr_forward(self, address_vec: np.ndarray) -> np.ndarray:
        events = sig_vec_to_events(address_vec, self.N_i)
        ft = rdlif_fire_times(self.W_addr, events, self.tau_r, self.tau_a,
                               self.threshold, t_window=self.t_window)
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
        ft = rdlif_fire_times(self.W_data, events, self.tau_r, self.tau_a,
                               self.threshold, t_window=self.t_window)
        top = np.argsort(ft)[:self.N_d]
        top = top[np.isfinite(ft[top])]
        out = np.zeros(self.D, dtype=np.float64)
        for rank, idx in enumerate(top):
            out[idx] = self.alpha ** rank
        return out
