"""
A faithful reimplementation of the WHEEL (firefly/spin) MODEL spiking
sequence machine, as actually built in Chapter 7 of the thesis ("Modelling
and simulating the complete spiking neural sequence machine").

Why the wheel model, not RDLIF: Chapter 6 explicitly investigates the
RDLIF model (the one in the IJCNN/ICANN/WIRN conference papers, and the
one used for the burst-stability simulations reproduced in the
sdm-nengo-2021-reproduction subfolder) and concludes it CANNOT be tuned to
reproduce the time-abstracted model's rank-order/significance-vector
behaviour exactly -- "implementing the temporal abstraction by the RDLIF
model... is not feasible" (p.125). The thesis then introduces the wheel
model specifically because it CAN be tuned to behave identically to the
abstract model, and states plainly: "we will use the wheel model to
simulate the complete spiking machine in the next chapter" (p.128). So
this -- not RDLIF -- is the neuron model that belongs in a faithful
"spiking version of the sequence machine."

Wheel model (thesis eq 6.10-6.13): each neuron has an activation/phase
that rises at a constant slope m once "active" (firefly-style), and jumps
instantaneously by (connection_weight * input_significance) whenever it
receives an input spike. It fires when activation crosses threshold Theta.
Because the jump and ramp dynamics are PIECEWISE LINEAR, the time-to-fire
has an exact closed-form solution given the sequence of incoming spike
events -- no numerical integration needed, which is what makes this
tractable to implement faithfully in this session.

    a_i(t) = m*(t - t0) + sum_{j: t_j <= t} w_ij * alpha_j      (eq 6.10)
    fires at first T where a_i(T) >= Theta                       (eq 6.11)

Architecture (thesis Fig 5.3 / 7.1): encoder (fixed lookup table) ->
context layer (combined model, fed both new input and delayed old
context) -> address decoder (fixed random weights) -> data store (a CMM,
the only layer with LEARNED weights, via max-Hebbian update) -> decoder
(lookup table, nearest-match).

Known, documented simplification: the data store's WRITE phase uses the
same instantaneous max-Hebbian rule as the abstract model (chapter 4/5),
not literal spike-timing-dependent plasticity over a continuous write
window (which would require modelling the "L" learning-signal timing
constraints from chapter 7.4.3 in more depth than is practical here).
Everything else -- encoding, context combination, address decoding, and
the READ/decode operation -- runs through genuine wheel-model spiking
dynamics with real closed-form fire-time computation, rank-order decode,
and significance-vector propagation between layers.
"""

import numpy as np


class WheelLayer:
    """A layer of M wheel-model neurons receiving a single upstream burst
    (a temporally ordered sequence of (source_index, significance) events)
    through a fixed weight matrix W [M, num_sources]. Computes exact
    closed-form fire times per thesis eq 6.10-6.13, then returns the
    output burst as the w earliest-firing neurons, each carrying their own
    significance (alpha^rank) for propagation to the next layer."""

    def __init__(self, M: int, slope: float = 1.0, threshold: float = 1.0):
        self.M = M
        self.slope = slope
        self.threshold = threshold

    def fire_times(self, W: np.ndarray, events: list) -> np.ndarray:
        """events: list of (source_idx, significance) in arrival order.
        W: [M, num_sources] connection weight matrix.
        Returns array of length M: fire time of each neuron (closed form).
        Synthetic unit-spaced arrival times (t_j = j) are used for the
        upstream events -- the wheel model's correctness depends on
        relative event ORDER and weighted jump magnitudes, not on the
        absolute inter-spike time gaps, so equal-spaced synthetic timing
        is a faithful, simplifying choice (consistent with the thesis's
        own "neurons in phase initially" / fixed-slope-m setup)."""
        if not events:
            return np.full(self.M, np.inf)

        cumulative = np.zeros(self.M)
        fired = np.full(self.M, False)
        fire_t = np.full(self.M, np.inf)

        t0 = 0.0
        for k, (src, alpha) in enumerate(events):
            t_k = float(k)  # synthetic unit-spaced arrival time
            # advance the ramp for all not-yet-fired neurons up to t_k,
            # check if any crossed threshold WITHIN the ramp segment
            # [t_{k-1}, t_k) before this jump is applied
            if k > 0:
                prev_t = float(k - 1)
                # a(t) = m*(t - t0) + cumulative   for t in [prev_t, t_k)
                # solve a(T) = Theta  ->  T = t0 + (Theta - cumulative)/m
                not_fired = ~fired
                with np.errstate(invalid="ignore"):
                    T = t0 + (self.threshold - cumulative) / self.slope
                crosses = not_fired & (T >= prev_t) & (T < t_k)
                fire_t[crosses] = T[crosses]
                fired |= crosses
            # apply the instantaneous jump from this event
            jump = W[:, src] * alpha
            cumulative = cumulative + jump
            # a neuron could already be over threshold the instant the
            # jump lands; record that as firing exactly at t_k
            just_over = (~fired) & (cumulative >= self.threshold)
            fire_t[just_over] = t_k
            fired |= just_over

        # after the last event, neurons keep ramping forever (no leak in
        # the wheel model) until they eventually cross threshold
        not_fired = ~fired
        last_t = float(len(events) - 1)
        T = last_t + (self.threshold - cumulative) / self.slope
        fire_t[not_fired] = T[not_fired]
        return fire_t

    def decode_burst(self, W: np.ndarray, events: list, k: int, alpha_decay: float = 0.9):
        """Run the layer and return the output burst: the k earliest-firing
        neurons, each tagged with rank-order significance alpha^rank, in
        firing order. This IS the rank-ordered N-of-M code for this layer's
        output, realised via actual spike timing rather than abstract
        vector math."""
        times = self.fire_times(W, events)
        order = np.argsort(times)[:k]
        order = order[np.isfinite(times[order])]
        out_events = [(int(idx), alpha_decay ** rank) for rank, idx in enumerate(order)]
        return out_events, times


def make_encoder_codebook(num_symbols: int, M: int, N: int, seed: int = 0):
    """Fixed lookup-table encoder: each symbol maps to a canonical ordered
    N-of-M burst (thesis: 'the encoder ... behaves like a lookup table')."""
    g = np.random.default_rng(seed)
    codebook = []
    for s in range(num_symbols):
        active = g.choice(M, size=N, replace=False)
        order = g.permutation(N)  # random firing order among the N active
        events = [(int(active[order[r]]), 0.9 ** r) for r in range(N)]
        codebook.append(events)
    return codebook


def decode_to_symbol(out_events, encoder_codebook):
    """Nearest-match decode: compare the FIRED NEURON SET (ignoring exact
    significance) to each symbol's canonical active-neuron set, pick the
    symbol with the largest overlap -- the spiking analogue of the
    cosine-similarity decode used in the abstract model."""
    fired_set = set(idx for idx, _ in out_events)
    best_sym, best_overlap = -1, -1
    for sym, events in enumerate(encoder_codebook):
        canon_set = set(idx for idx, _ in events)
        overlap = len(fired_set & canon_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sym = sym
    return best_sym


class WheelSpikingSequenceMachine:
    """The full spiking sequence machine: encoder -> context (combined
    model, wheel) -> address decoder (wheel) -> data store / CMM (wheel,
    learned weights) -> decoder, following the thesis's 3-step on-line
    framework (write with OLD context, update context, read/predict with
    NEW context) exactly as in chapter 5 eq 5.1-5.3."""

    def __init__(self, num_symbols: int, M: int = 64, N: int = 6,
                 ctx_dim: int = 96, ctx_k: int = 10,
                 addr_dim: int = 128, addr_k: int = 10,
                 Lambda: float = 0.5, seed: int = 0):
        self.encoder_codebook = make_encoder_codebook(num_symbols, M, N, seed=seed)
        self.M, self.N = M, N
        self.ctx_dim, self.ctx_k = ctx_dim, ctx_k
        self.addr_dim, self.addr_k = addr_dim, addr_k
        self.Lambda = Lambda

        g = np.random.default_rng(seed + 1)
        self.context_layer = WheelLayer(ctx_dim, slope=1.0, threshold=1000.0)
        # P1: old-context (ctx_dim) -> context layer; P2: input (M) -> context layer
        self.P1 = g.random((ctx_dim, ctx_dim))
        self.P2 = g.random((ctx_dim, M))

        self.addr_layer = WheelLayer(addr_dim, slope=1.0, threshold=1000.0)
        self.addr_weights = g.random((addr_dim, ctx_dim))  # fixed

        self.data_store_layer = WheelLayer(M, slope=1.0, threshold=1000.0)
        self.W_data = np.zeros((M, addr_dim))  # LEARNED via max-Hebbian

        self.context_events = []  # current context burst, starts empty

    def _addr_decode(self, ctx_events):
        return self.addr_layer.decode_burst(self.addr_weights, ctx_events,
                                             k=self.addr_k)

    def step(self, symbol: int):
        input_events = self.encoder_codebook[symbol]

        # --- step 1 (thesis eq 5.1): write association at OLD context ---
        addr_events_old, _ = self._addr_decode(self.context_events)
        addr_vec = np.zeros(self.addr_dim)
        for idx, _ in addr_events_old:
            addr_vec[idx] = 1.0
        input_vec = np.zeros(self.M)
        for idx, _ in input_events:
            input_vec[idx] = 1.0
        self.W_data = np.maximum(self.W_data, np.outer(input_vec, addr_vec))

        # --- step 2 (thesis eq 5.2): combined-model context update ---
        # old context contributes scaled by Lambda (via P1), input by (1-Lambda) (via P2)
        # build a combined weight view: P1 columns for context indices,
        # P2 columns for input indices, stacked as one "source space"
        ctx_n = self.ctx_dim
        W_combined = np.concatenate([self.P1, self.P2], axis=1)
        events_remapped = []
        for idx, alpha in self.context_events:
            events_remapped.append((idx, alpha * self.Lambda))  # P1 columns: 0..ctx_dim-1
        for idx, alpha in input_events:
            events_remapped.append((ctx_n + idx, alpha * (1 - self.Lambda)))  # P2 columns offset
        new_context_events, _ = self.context_layer.decode_burst(
            W_combined, events_remapped, k=self.ctx_k)
        self.context_events = new_context_events

        # --- step 3 (thesis eq 5.3): predict by reading with NEW context ---
        addr_events_new, _ = self._addr_decode(self.context_events)
        out_events, _ = self.data_store_layer.decode_burst(
            self.W_data, addr_events_new, k=self.N)
        prediction = decode_to_symbol(out_events, self.encoder_codebook)
        return prediction

    def reset(self):
        self.context_events = []
        self.W_data = np.zeros((self.M, self.addr_dim))

    def run_sequence(self, seq):
        self.reset()
        preds = []
        for sym in seq:
            preds.append(self.step(sym))
        return preds
