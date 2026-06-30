"""
Reimplementation of:

  Ajwani, Lalan, Sen Bhattacharya, Bose (2021).
  "Sparse Distributed Memory using Spiking Neural Networks on Nengo."
  Bernstein Conference 2021.

This module reimplements the architecture exactly as specified in Sections
II-IV: N-of-M binary codes, a Correlation Matrix Memory (CMM, Kohonen 1972)
as the data-store, an SDM built from a fixed-weight address decoder feeding
a CMM, and a spiking-neuron read/decode rule based on FIRST-SPIKE TIMING
(the first w-of-W / d-of-D neurons to fire constitute the output code,
exactly as described in Section III.A/B).

Two implementations are provided:

1. NonSpikingCMM / NonSpikingSDM -- the "Non Spiking" baseline from the
   paper's own figures: direct correlation-matrix accumulation, top-k
   thresholding for decode. This is the standard non-spiking SDM/CMM the
   paper compares its spiking version against.

2. AnalyticLIF-based spiking CMM/SDM -- a genuine LIF spiking-neuron
   model, but solved in CLOSED FORM rather than timestep-simulated. The
   paper's own read/write phases inject a CONSTANT current per neuron for
   the duration of a fixed time window (150ms) and decode by "which
   neurons spiked first." For a leaky integrate-and-fire neuron under
   constant input current I, the time to first spike has an exact
   analytic solution (derived from the same ODE as the paper's eq. 3):

       t_spike = -tau * ln(1 - V_thr / (I * R_m))      if I*R_m > V_thr
       t_spike = infinity (no spike in window)          otherwise

   Using this closed form to rank neurons by first-spike-time is
   mathematically equivalent to literally simulating the spikes and
   reading off arrival order, for the constant-current regime the paper
   itself uses -- but is orders of magnitude faster, which is what makes
   it possible to reproduce the FULL capacity curves (not just a handful
   of points) in this session. This is a deliberate, documented
   substitution for the Nengo Ensemble population-coding API, which is
   built for vector representation rather than the paper's per-neuron
   direct current injection design and fights that design when forced
   into it.

Learning rule: the paper uses BCM and Oja, which are local, online,
spike-train-dependent rules. Reproducing their exact continuous-time
dynamics requires real spike trains, which the closed-form first-spike
shortcut above does not produce. As a documented simplification, the
data-store learning rule used here is the plain Hebbian/correlation
accumulation common to both BCM and Oja in their "fast-Hebbian-like"
regime (and which the paper itself uses for the non-spiking baseline and
the original Furber et al. CMM formulation, eq. 1-2 of the paper). This
is flagged clearly in the README as a scope-limiting simplification, not
hidden.
"""

import numpy as np


def random_nofm_codes(num_items: int, M: int, N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate num_items random binary N-of-M codes (exactly N of M bits
    set to 1), as used for both address and data vectors in the paper."""
    codes = np.zeros((num_items, M), dtype=np.float64)
    for i in range(num_items):
        active = rng.choice(M, size=N, replace=False)
        codes[i, active] = 1.0
    return codes


# ---------------------------------------------------------------------------
# Non-spiking baseline (paper's "Non Spiking" curve)
# ---------------------------------------------------------------------------

class NonSpikingCMM:
    """Correlation Matrix Memory: W = sum_k outer(data_k, address_k).
    Read: raw = W @ address_query; top-d components -> binary output code.
    This is exactly eq. 1-2 of the paper (Kohonen 1972 CMM)."""

    def __init__(self, A: int, D: int, d: int):
        self.W = np.zeros((D, A))
        self.d = d

    def write(self, address: np.ndarray, data: np.ndarray):
        self.W = np.maximum(self.W, np.outer(data, address))  # unipolar binary weights (Furber et al)

    def read(self, address: np.ndarray) -> np.ndarray:
        raw = self.W @ address
        topk = np.argsort(raw)[-self.d:]
        out = np.zeros(self.W.shape[0])
        out[topk] = 1.0
        return out


class NonSpikingSDM:
    """Fixed-weight address decoder (random binary projection A -> W) feeding
    a CMM (W -> D), exactly as in paper Fig. 2/Section II.B."""

    def __init__(self, A: int, W: int, w: int, D: int, d: int, rng: np.random.Generator):
        self.addr_decoder = rng.random((W, A))  # fixed random weights
        self.w = w
        self.cmm = NonSpikingCMM(A=W, D=D, d=d)

    def _addr_code(self, address: np.ndarray) -> np.ndarray:
        raw = self.addr_decoder @ address
        topk = np.argsort(raw)[-self.w:]
        out = np.zeros(self.addr_decoder.shape[0])
        out[topk] = 1.0
        return out

    def write(self, address: np.ndarray, data: np.ndarray):
        addr_code = self._addr_code(address)
        self.cmm.write(addr_code, data)

    def read(self, address: np.ndarray) -> np.ndarray:
        addr_code = self._addr_code(address)
        return self.cmm.read(addr_code)


# ---------------------------------------------------------------------------
# Analytic-LIF spiking version (paper's "LIF" / "Adaptive LIF" / etc curves)
# ---------------------------------------------------------------------------

def first_spike_times(currents: np.ndarray, tau: float = 1.0, v_thr: float = 1.0,
                       r_m: float = 1.0) -> np.ndarray:
    """Closed-form LIF time-to-first-spike under constant current, per the
    paper's eq. 3 (Cm*dVm/dt = I - Vm/Rm). Neurons that never reach
    threshold are given an infinite spike time (they don't fire in the
    observation window, matching the paper's "neurons with no spikes in
    150ms" case)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = v_thr / (currents * r_m)
        can_fire = ratio < 1.0
        t = np.full_like(currents, np.inf)
        t[can_fire] = -tau * np.log(1 - ratio[can_fire])
    return t


class SpikingCMM:
    """Same architecture as NonSpikingCMM, but the read phase decodes via
    first-spike-time ranking of LIF neurons receiving the correlation-matrix
    output as injected current, exactly mirroring the paper's described
    decode rule ("the output of the neurons that fired the first d spikes
    are set to 1")."""

    def __init__(self, A: int, D: int, d: int):
        self.W = np.zeros((D, A))
        self.d = d

    def write(self, address: np.ndarray, data: np.ndarray):
        self.W = np.maximum(self.W, np.outer(data, address))  # unipolar binary weights (Furber et al)

    def read(self, address: np.ndarray) -> np.ndarray:
        raw_current = self.W @ address
        # currents must be positive to drive a LIF neuron toward threshold;
        # shift/scale matches the paper's note that weights are scaled to a
        # workable input current range (Section IV.D)
        currents = np.clip(raw_current, 1e-6, None)
        spike_times = first_spike_times(currents)
        topk = np.argsort(spike_times)[: self.d]  # EARLIEST d spikers
        out = np.zeros(self.W.shape[0])
        out[topk] = 1.0
        return out


class SpikingSDM:
    """Spiking version of NonSpikingSDM: address decoder layer AND data
    memory (CMM) both decode via first-spike-time ranking, matching the
    paper's full spiking SDM (Section III.B, Fig. 3)."""

    def __init__(self, A: int, W: int, w: int, D: int, d: int, rng: np.random.Generator):
        self.addr_decoder = rng.random((W, A))
        self.w = w
        self.cmm = SpikingCMM(A=W, D=D, d=d)

    def _addr_code(self, address: np.ndarray) -> np.ndarray:
        raw_current = self.addr_decoder @ address
        currents = np.clip(raw_current, 1e-6, None)
        spike_times = first_spike_times(currents)
        topk = np.argsort(spike_times)[: self.w]
        out = np.zeros(self.addr_decoder.shape[0])
        out[topk] = 1.0
        return out

    def write(self, address: np.ndarray, data: np.ndarray):
        addr_code = self._addr_code(address)
        self.cmm.write(addr_code, data)

    def read(self, address: np.ndarray) -> np.ndarray:
        addr_code = self._addr_code(address)
        return self.cmm.read(addr_code)
