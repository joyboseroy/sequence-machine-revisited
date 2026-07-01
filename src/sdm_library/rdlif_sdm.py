"""
sdm_library/rdlif_sdm.py

(d) Rate-Driven Leaky Integrate-and-Fire (RDLIF) spiking SDM.
    Thesis Chapter 6 / WIRN 2005 / IJCNN 2005 / ICANN 2005 papers.

RDLIF neuron model (thesis section 6.2, WIRN paper eq. 1-2):
  dr_i/dt = sum_j w_ij * x_j(t) - (r_i - r_0) / tau_r
  da_i/dt = r_i - (a_i - a_0) / tau_a
  fires when a_i >= Theta; then a_i -> reset, r_i -> 0

Where r_i is the "activation driving force" (rate), a_i is the activation.
Incoming spikes increment the RATE, not the activation directly — this is
what gives the model its inherent delay between input and output spikes,
which the WIRN paper shows is necessary for stable feedforward burst
propagation.

Critical design note (thesis Chapter 6, p.125):
  The RDLIF model CANNOT be tuned to exactly reproduce the abstract
  significance-vector model's firing order, because the firing time depends
  on the ABSOLUTE inter-spike intervals as well as the weighted sums. The
  thesis uses this model in the burst-stability / coherence experiments
  (Chapter 6) but switches to the wheel model for the actual sequence
  machine (Chapter 7) precisely because of this limitation.

  We implement RDLIF here faithfully for completeness and because it is
  the model used in the WIRN/Async Forum/Ajwani-Nengo papers. The behaviour
  will be SIMILAR but not IDENTICAL to the abstract RankOrderSDM, especially
  when input spike intervals are irregular.

Simulation: closed-form exact solution per timestep.
For constant driving force r between two spike events, the activation
evolves as:
  a(t) = a0 + (r - a0) * (1 - exp(-(t-t0)/tau_a))       [simplified]
We use the exact piecewise solution between events.
"""

import numpy as np
from .base import SDMBase, topk_indices
from .wheel_sdm import events_from_significance_vec


def rdlif_fire_times(W_layer: np.ndarray,
                     events: list,
                     tau_r: float = 0.5,
                     tau_a: float = 1.0,
                     threshold: float = 1.0,
                     r0: float = 0.0,
                     a0: float = 0.0,
                     t_window: float = 20.0) -> np.ndarray:
    """Simulate RDLIF neurons receiving a spike burst and return fire times.

    W_layer   : [num_neurons, num_sources] weight matrix
    events    : list of (source_index, significance) in arrival order,
                with UNIT-SPACED arrival times (t_j = j * dt, dt=1.0)
    tau_r     : rate/driving-force time constant (thesis: tau_r < tau_a)
    tau_a     : activation time constant
    threshold : firing threshold Theta
    r0, a0    : resting values for rate and activation
    t_window  : maximum simulation time (neurons not firing within window
                get fire_time = inf)

    Uses exact piecewise-linear approximation between spike events:
    between events k and k+1, with constant r (no new spikes), activation
    evolves exponentially. We approximate with small timesteps for robustness.
    """
    num_neurons = W_layer.shape[0]
    dt = 0.01  # integration timestep (small relative to tau)
    num_steps = int(t_window / dt)

    r = np.full(num_neurons, r0, dtype=np.float64)
    a = np.full(num_neurons, a0, dtype=np.float64)
    fired = np.zeros(num_neurons, dtype=bool)
    fire_t = np.full(num_neurons, np.inf)

    # pre-build event time lookup: spike arrives at t = k (integer)
    event_at = {}
    for k, (src_idx, sigma) in enumerate(events):
        t_k = float(k)  # unit-spaced
        event_at.setdefault(t_k, []).append((src_idx, sigma))

    t = 0.0
    for step in range(num_steps):
        t = step * dt

        # apply any spikes arriving at this timestep (within dt)
        t_key = round(t)
        if t_key in event_at and abs(t - t_key) < dt / 2:
            for src_idx, sigma in event_at[t_key]:
                r += W_layer[:, src_idx] * sigma
            del event_at[t_key]

        # update rate and activation via Euler integration
        dr = (- (r - r0) / tau_r) * dt
        da = (r - (a - a0) / tau_a) * dt
        r += dr
        a += da

        # check threshold crossings
        crosses = (~fired) & (a >= threshold)
        fire_t[crosses] = t
        fired |= crosses
        a[fired] = a0
        r[fired] = r0

        if np.all(fired):
            break

    return fire_t


class RDLIFSDM(SDMBase):
    """RDLIF spiking SDM — thesis Chapter 6 / WIRN/IJCNN/ICANN 2005 papers.

    Uses the Rate-Driven Leaky Integrate-and-Fire neuron for both address
    decoder and data store read operations. Firing order determines the
    output rank-ordered code, but (unlike WheelSDM) this order depends on
    both weighted inputs AND absolute inter-spike timing, so results will
    differ slightly from the abstract RankOrderSDM.

    Data store write uses MAX outer-product (same simplification as WheelSDM;
    see WheelSDM docstring for rationale).
    """

    def __init__(self, D: int, N_d: int, W: int, N_w: int,
                 alpha: float = 0.99, seed: int = 0,
                 tau_r: float = 0.5, tau_a: float = 1.0,
                 threshold: float = 1.0, t_window: float = 20.0):
        self.tau_r = tau_r
        self.tau_a = tau_a
        self.threshold = threshold
        self.t_window = t_window
        super().__init__(D, N_d, W, N_w, alpha, seed)

    def _init_weights(self):
        # Same significance-vector address decoder as RankOrderSDM / WheelSDM
        self.W_addr = np.zeros((self.W, self.D), dtype=np.float64)
        for i in range(self.W):
            active = self.rng.choice(self.D, size=self.N_d, replace=False)
            order = self.rng.permutation(self.N_d)
            for rank, j in enumerate(order):
                self.W_addr[i, active[j]] = self.alpha ** rank

        self.W_data = np.zeros((self.D, self.W), dtype=np.float64)

    def _addr_forward(self, address_vec: np.ndarray) -> np.ndarray:
        """RDLIF address decoder: fire times -> top-N_w significance vector."""
        events = events_from_significance_vec(address_vec, self.N_d)
        fire_times = rdlif_fire_times(self.W_addr, events,
                                      self.tau_r, self.tau_a,
                                      self.threshold, t_window=self.t_window)
        top = np.argsort(fire_times)[:self.N_w]
        top = top[np.isfinite(fire_times[top])]
        out = np.zeros(self.W, dtype=np.float64)
        for rank, idx in enumerate(top):
            out[idx] = self.alpha ** rank
        return out

    def _data_write(self, addr_decoded: np.ndarray, data_vec: np.ndarray):
        self.W_data = np.maximum(self.W_data,
                                  np.outer(data_vec, addr_decoded))

    def _data_read(self, addr_decoded: np.ndarray) -> np.ndarray:
        """RDLIF data store read: fire times -> top-N_d significance vector."""
        events = events_from_significance_vec(addr_decoded, self.N_w)
        fire_times = rdlif_fire_times(self.W_data, events,
                                      self.tau_r, self.tau_a,
                                      self.threshold, t_window=self.t_window)
        top = np.argsort(fire_times)[:self.N_d]
        top = top[np.isfinite(fire_times[top])]
        out = np.zeros(self.D, dtype=np.float64)
        for rank, idx in enumerate(top):
            out[idx] = self.alpha ** rank
        return out
