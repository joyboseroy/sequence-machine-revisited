"""
sdm_library/standard_sdm.py

(a) Standard binary N-of-M SDM — Kanerva / Furber et al. version,
    thesis Chapter 3 (sections 3.3.4-3.3.5).

Three distinct N-of-M codes in play (thesis Figure 3.2):
  i-of-A : input address (e.g. 11-of-256)  -> N_i, D
  a-of-A : address decoder weights          -> N_a, D  (N_a != N_i in general)
  w-of-W : address decoder output           -> N_w, W
  d-of-D : data code (often = i-of-A)       -> N_d, D

Each of the W address decoder neurons has exactly N_a non-zero binary
weights. The dot product of an i-of-A input with an a-of-A weight vector
gives the number of matching active bits. Top-N_w decoders fire.

Learning: logical-OR update on data store (binary weights).
Decode:   L-max (top-N_d) thresholding on dot-product output.

Thesis eq. 3.8-3.11, Figure 3.2.
"""

import numpy as np
from .base import SDMBase, topk_indices


class StandardSDM(SDMBase):

    def _init_weights(self):
        # Address decoder: W neurons, each with N_a binary weights out of D
        # Thesis p.78: "weights are set to a random a-of-A code"
        # N_a = a (separate from N_i = i, the input code sparsity)
        self.W_addr = np.zeros((self.W, self.D), dtype=np.float64)
        for i in range(self.W):
            active = self.rng.choice(self.D, size=self.N_a, replace=False)
            self.W_addr[i, active] = 1.0
        # Data store: binary weights, initialised to 0
        self.W_data = np.zeros((self.D, self.W), dtype=np.float64)

    def _addr_forward(self, address_vec: np.ndarray) -> np.ndarray:
        """Dot product of i-of-A input with a-of-A decoder weights.
        Select top-N_w decoders -> binary w-of-W output.
        Thesis eq. 3.10: w = Theta(A*x, tA)"""
        activations = self.W_addr @ address_vec
        out = np.zeros(self.W, dtype=np.float64)
        out[topk_indices(activations, self.N_w)] = 1.0
        return out

    def _data_write(self, addr_decoded: np.ndarray, data_vec: np.ndarray):
        """Logical-OR update: W_data = W_data OR outer(data, addr_decoded).
        Thesis eq. 3.9: D = D ⊕ y^T"""
        self.W_data = np.maximum(self.W_data,
                                  np.outer(data_vec, addr_decoded))

    def _data_read(self, addr_decoded: np.ndarray) -> np.ndarray:
        """Dot product then L-max (top-N_d) decode.
        Thesis eq. 3.11: y = L(D^T * w, d)"""
        raw = self.W_data @ addr_decoded
        out = np.zeros(self.D, dtype=np.float64)
        out[topk_indices(raw, self.N_d)] = 1.0
        return out
