"""
sdm_library/rankorder_sdm.py

(b) Rank-ordered N-of-M SDM — thesis Chapter 4 (section 4.2).

Same three-layer structure as StandardSDM (Figure 3.2) but with two changes
(thesis p.85):
  1. Symbols encoded as significance vectors instead of binary N-of-M.
     Active components carry weights 1, alpha, alpha^2, ... by firing order.
  2. Data store uses real-valued weights and MAX learning rule instead of
     binary weights and logical-OR.

Address decoder weights are also significance vectors (a-of-A with rank-
order significance), so the dot product is a significance-weighted inner
product. The N_a parameter controls how many of D inputs each decoder
connects to (the 'a' in a-of-A), separate from N_i (the input code sparsity).

Learning rule: w_ij = max(w_ij, sigma_i * sigma_j)  [thesis Fig 4.2]
Decode: top-N_d by activation, re-encoded as significance vector.

Thesis reference: Chapter 4, section 4.2, Figure 4.2
Default parameters (section 4.3.5): N_i=N_d=11, N_a=20, W=4096, N_w=16
"""

import numpy as np
from .base import SDMBase, topk_indices


class RankOrderSDM(SDMBase):

    def _init_weights(self):
        # Address decoder: W neurons, each with N_a significance-weighted
        # connections out of D inputs.
        # Thesis section 4.3.3: "each address decoder ... non-zero weights
        # for exactly d [= N_a here] out of D input neurons ... set as per
        # the [1, alpha, alpha^2..] significance vector"
        self.W_addr = np.zeros((self.W, self.D), dtype=np.float64)
        for i in range(self.W):
            active = self.rng.choice(self.D, size=self.N_a, replace=False)
            order = self.rng.permutation(self.N_a)
            for rank, j in enumerate(order):
                self.W_addr[i, active[j]] = self.alpha ** rank
        # Data store: real-valued, initialised to 0
        self.W_data = np.zeros((self.D, self.W), dtype=np.float64)

    def _addr_forward(self, address_vec: np.ndarray) -> np.ndarray:
        """Significance-weighted dot product, select top-N_w decoders,
        re-encode output as a w-of-W significance vector."""
        activations = self.W_addr @ address_vec
        top = topk_indices(activations, self.N_w)
        top_sorted = top[np.argsort(activations[top])[::-1]]
        out = np.zeros(self.W, dtype=np.float64)
        for rank, idx in enumerate(top_sorted):
            out[idx] = self.alpha ** rank
        return out

    def _data_write(self, addr_decoded: np.ndarray, data_vec: np.ndarray):
        """MAX outer product: w_ij = max(w_ij, sigma_i * sigma_j).
        Thesis Figure 4.2."""
        self.W_data = np.maximum(self.W_data,
                                  np.outer(data_vec, addr_decoded))

    def _data_read(self, addr_decoded: np.ndarray) -> np.ndarray:
        """Significance-weighted dot product then top-N_d re-encoded as
        significance vector."""
        raw = self.W_data @ addr_decoded
        top = topk_indices(raw, self.N_d)
        top_sorted = top[np.argsort(raw[top])[::-1]]
        out = np.zeros(self.D, dtype=np.float64)
        for rank, idx in enumerate(top_sorted):
            out[idx] = self.alpha ** rank
        return out
