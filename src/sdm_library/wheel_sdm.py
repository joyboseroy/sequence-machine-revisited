"""
sdm_library/wheel_sdm.py

(c) Wheel-model spiking SDM — thesis Chapters 6 and 7.

The wheel (firefly/spin) neuron model, thesis section 6.4:
  - Each neuron has a phase that rises at constant slope m (the "spin rate")
  - On receiving an input spike with significance sigma, the phase jumps
    instantaneously by (connection_weight * sigma)
  - The neuron fires when its phase crosses threshold Theta
  - After firing, phase resets to 0

This is the model the thesis ACTUALLY uses for the full spiking
sequence machine (Chapter 7), chosen specifically because it CAN be
tuned to reproduce the time-abstracted significance-vector model exactly.
(Chapter 6 shows RDLIF cannot achieve this.) See thesis p.127-128:
"the wheel model ... it is easier to tune it to behave precisely as we need."

Key property: under the right threshold (Theta >> max_possible_jump_sum),
firing order is determined entirely by cumulative weighted input, which is
exactly the dot-product calculation of the abstract model. This gives a
direct, exact correspondence between the spiking and non-spiking models.

Closed-form fire time (thesis eq. 6.10-6.13):
  a(t) = m*(t - t_reset) + sum_{j: t_j<=t} w_ij * sigma_j
  fires at first T where a(T) >= Theta

We use synthetic unit-spaced arrival times for incoming spikes (t_j = j),
which is consistent with the thesis's own "neurons in phase initially /
fixed slope m" assumption -- only relative arrival ORDER and significance
magnitudes affect firing order under the high-threshold regime.
"""

import numpy as np
from .base import SDMBase, topk_indices


def wheel_fire_times(W_layer: np.ndarray,
                     events: list,
                     slope: float = 1.0,
                     threshold: float = 1000.0) -> np.ndarray:
    """Compute closed-form fire times for all neurons in a layer.

    W_layer : [num_neurons, num_sources] weight matrix
    events  : list of (source_index, significance) in arrival order
    slope   : constant phase increase rate m (thesis eq. 6.10)
    threshold: Theta, must be >> max possible single-spike jump for
               the high-threshold regime where fire order = dot-product order

    Returns array of length num_neurons: fire time of each neuron.
    Neurons that never accumulate enough to fire get inf.
    """
    num_neurons = W_layer.shape[0]
    cumulative = np.zeros(num_neurons)
    fired = np.zeros(num_neurons, dtype=bool)
    fire_t = np.full(num_neurons, np.inf)

    for k, (src_idx, sigma) in enumerate(events):
        t_k = float(k)  # synthetic unit-spaced arrival times

        # check if any neuron crossed threshold during ramp since last event
        if k > 0:
            prev_t = float(k - 1)
            # solve m*(T - 0) + cumulative = threshold
            # (t_reset=0 since we track cumulative separately)
            T = (threshold - cumulative) / slope
            crosses = (~fired) & (T >= prev_t) & (T < t_k)
            fire_t[crosses] = T[crosses]
            fired |= crosses

        # apply instantaneous jump from this spike
        cumulative += W_layer[:, src_idx] * sigma

        # check if jump pushed any over threshold immediately
        over = (~fired) & (cumulative >= threshold)
        fire_t[over] = t_k
        fired |= over

    # after last event: remaining neurons fire at their ramp-crossing time
    not_fired = ~fired
    if np.any(not_fired):
        T = (threshold - cumulative[not_fired]) / slope
        fire_t[not_fired] = float(len(events)) + T

    return fire_t


def events_from_significance_vec(vec: np.ndarray, N: int) -> list:
    """Convert a significance vector to an ordered spike event list.
    The N largest components become (index, value) events sorted by
    DESCENDING significance (i.e. the neuron with weight 1.0 fires first).
    """
    top = topk_indices(vec, N)
    top_sorted = top[np.argsort(vec[top])[::-1]]
    return [(int(idx), float(vec[idx])) for idx in top_sorted]


class WheelSDM(SDMBase):
    """Rank-ordered N-of-M SDM using wheel-model spiking neurons.

    Encodes symbols as significance vectors (same as RankOrderSDM).
    Address decoder and data read both computed via wheel-model fire times.
    Data store write uses the same MAX outer-product rule as RankOrderSDM
    (documented simplification: full STDP over a continuous write window
    would require real spike trains from BOTH sides simultaneously; the
    max-Hebbian rule is the documented approximation used throughout the
    thesis's abstract-model derivations in Chapter 4/5).
    """

    def __init__(self, D: int, N_d: int, W: int, N_w: int,
                 alpha: float = 0.99, seed: int = 0,
                 slope: float = 1.0, threshold: float = 1000.0):
        self.slope = slope
        self.threshold = threshold
        super().__init__(D, N_d, W, N_w, alpha, seed)

    def _init_weights(self):
        # Address decoder: significance-vector weights (same as RankOrderSDM)
        self.W_addr = np.zeros((self.W, self.D), dtype=np.float64)
        for i in range(self.W):
            active = self.rng.choice(self.D, size=self.N_d, replace=False)
            order = self.rng.permutation(self.N_d)
            for rank, j in enumerate(order):
                self.W_addr[i, active[j]] = self.alpha ** rank

        # Data store: real-valued, initialised to 0
        self.W_data = np.zeros((self.D, self.W), dtype=np.float64)

    def _addr_forward(self, address_vec: np.ndarray) -> np.ndarray:
        """Run address-decoder neurons via wheel model.
        Top N_w earliest-firing neurons form the output significance vector."""
        events = events_from_significance_vec(address_vec, self.N_d)
        fire_times = wheel_fire_times(self.W_addr, events,
                                      self.slope, self.threshold)
        # earliest N_w fire = output burst
        top = np.argsort(fire_times)[:self.N_w]
        top = top[np.isfinite(fire_times[top])]
        out = np.zeros(self.W, dtype=np.float64)
        # assign significance by fire-time rank (earliest = rank 0 = weight 1)
        for rank, idx in enumerate(top):
            out[idx] = self.alpha ** rank
        return out

    def _data_write(self, addr_decoded: np.ndarray, data_vec: np.ndarray):
        """MAX outer-product update (same as RankOrderSDM).
        See module docstring for the documented simplification note."""
        self.W_data = np.maximum(self.W_data,
                                  np.outer(data_vec, addr_decoded))

    def _data_read(self, addr_decoded: np.ndarray) -> np.ndarray:
        """Run data-store neurons via wheel model.
        Top N_d earliest-firing neurons form the output significance vector."""
        events = events_from_significance_vec(addr_decoded, self.N_w)
        # W_data is [D, W]: D data neurons, each connected to W addr-decoder neurons
        fire_times = wheel_fire_times(self.W_data, events,
                                      self.slope, self.threshold)
        top = np.argsort(fire_times)[:self.N_d]
        top = top[np.isfinite(fire_times[top])]
        out = np.zeros(self.D, dtype=np.float64)
        for rank, idx in enumerate(top):
            out[idx] = self.alpha ** rank
        return out
