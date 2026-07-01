"""
sdm_library/standard_sdm.py

(a) Standard binary N-of-M SDM — Kanerva / Furber et al. version,
    as described in thesis Chapter 3 (section 3.3.4 and 3.3.5).

Architecture:
  - Symbols encoded as binary N-of-M vectors (exactly N of D bits set)
  - Address decoder: fixed random binary a-of-A weight matrix; output is
    a binary w-of-W vector (top-w threshold from dot product)
  - Data store: binary weights, updated by logical-OR (Willshaw/Furber rule,
    thesis eq. 3.9: D = D OR outer(data, addr_decoded))
  - Read: dot product then L-max (top-N_d) thresholding

Thesis reference: Chapter 3, section 3.3.5, eq. 3.8-3.11
Default parameters (Chapter 4 section 4.3.5): d=11, D=256, w=16, W=4096
"""

import numpy as np
from .base import SDMBase, topk_indices


class StandardSDM(SDMBase):
    """Binary N-of-M SDM (Kanerva / Furber et al., thesis Chapter 3).

    Symbols are binary N-of-M vectors. Address decoder weights are fixed
    random binary (a-of-A per decoder). Data store uses logical-OR update
    (binary weights, 0 or 1 only). Read decodes by L-max thresholding on
    the dot-product output.
    """

    def _init_weights(self):
        # Address decoder: W decoders, each connected to N_d of D inputs
        # via a random binary a-of-A weight vector.
        # Thesis: "the weight matrix ... set to a random a-of-A code"
        self.W_addr = np.zeros((self.W, self.D), dtype=np.float64)
        for i in range(self.W):
            active = self.rng.choice(self.D, size=self.N_d, replace=False)
            self.W_addr[i, active] = 1.0

        # Data store: binary weights, initialised to 0
        # Thesis: "initially the data memory ... weights are set to 0"
        self.W_data = np.zeros((self.D, self.W), dtype=np.float64)

    def _addr_forward(self, address_vec: np.ndarray) -> np.ndarray:
        """Dot product then top-N_w threshold -> binary w-of-W output.
        Thesis eq. 3.10: w = Theta(A*x, tA)"""
        activations = self.W_addr @ address_vec
        out = np.zeros(self.W, dtype=np.float64)
        top = topk_indices(activations, self.N_w)
        out[top] = 1.0
        return out

    def _data_write(self, addr_decoded: np.ndarray, data_vec: np.ndarray):
        """Logical-OR update: W_data = W_data OR outer(data, addr_decoded).
        Thesis eq. 3.9: D = D OR y^T (where y is addr_decoded, D is data)
        For binary weights this is equivalent to: set bit if EITHER was set."""
        self.W_data = np.maximum(self.W_data,
                                  np.outer(data_vec, addr_decoded))

    def _data_read(self, addr_decoded: np.ndarray) -> np.ndarray:
        """Dot product with data store, then L-max (top-N_d) decode.
        Thesis eq. 3.11: y = L(D^T * w, d)"""
        raw = self.W_data @ addr_decoded
        out = np.zeros(self.D, dtype=np.float64)
        top = topk_indices(raw, self.N_d)
        out[top] = 1.0
        return out
