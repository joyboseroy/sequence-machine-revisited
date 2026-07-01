"""
sdm_library/rankorder_sdm.py

(b) Rank-ordered N-of-M SDM — thesis Chapter 4 (section 4.2).

The two changes from the standard binary SDM (thesis p.85):
  1. Symbols encoded as ordered N-of-M significance vectors, not binary.
     Active components carry weights 1, alpha, alpha^2, ... by firing order.
  2. Data store uses real-valued weights and the MAX learning rule instead
     of binary weights and logical-OR.

Learning rule (thesis Fig 4.2, eq. on p.87):
    w_ij = max(w_ij, sigma_i * sigma_j)
where sigma_i is the significance of the i-th address decoder output bit
and sigma_j is the significance of the j-th data bit.

Address decoder weights are also significance vectors (not binary), so
the dot product naturally produces a significance-weighted activation.

Thesis reference: Chapter 4, section 4.2, Figure 4.2
Default parameters (section 4.3.5): d=11, D=256, w=16, W=4096, alpha=0.99
"""

import numpy as np
from .base import SDMBase, topk_indices


class RankOrderSDM(SDMBase):
    """Rank-ordered N-of-M SDM with significance vectors, thesis Chapter 4.

    Symbols are significance vectors (alpha^rank at active positions, 0
    elsewhere). Address decoder has real-valued significance weights.
    Data store uses MAX of outer product as learning rule (real-valued
    weights). Read decodes by sorting on dot-product activations then
    re-encoding the top-N_d as a significance vector.
    """

    def _init_weights(self):
        # Address decoder: W decoders with significance vectors
        # Thesis section 4.3.3: "each address decoder ... non-zero weights
        # for exactly d out of D input neurons ... set as per the [1,alpha,alpha^2..]
        # significance vector. Exactly d random values ... sorted to find
        # a random order for the input weights."
        self.W_addr = np.zeros((self.W, self.D), dtype=np.float64)
        for i in range(self.W):
            active = self.rng.choice(self.D, size=self.N_d, replace=False)
            # random ordering within the N_d active neurons
            order = self.rng.permutation(self.N_d)
            for rank, j in enumerate(order):
                self.W_addr[i, active[j]] = self.alpha ** rank

        # Data store: real-valued weights, initialised to 0
        self.W_data = np.zeros((self.D, self.W), dtype=np.float64)

    def _addr_forward(self, address_vec: np.ndarray) -> np.ndarray:
        """Dot product (significance-weighted) then top-N_w selection.
        Returns a significance vector: top-N_w indices get weights
        alpha^0, alpha^1, ... by their activation rank."""
        activations = self.W_addr @ address_vec
        top = topk_indices(activations, self.N_w)
        # sort selected indices by descending activation to assign significance
        top_sorted = top[np.argsort(activations[top])[::-1]]
        out = np.zeros(self.W, dtype=np.float64)
        for rank, idx in enumerate(top_sorted):
            out[idx] = self.alpha ** rank
        return out

    def _data_write(self, addr_decoded: np.ndarray, data_vec: np.ndarray):
        """MAX outer product update: w_ij = max(w_ij, sigma_i * sigma_j).
        Thesis Fig 4.2: w_ij = max(w_ij, sigma_i * sigma_j)"""
        self.W_data = np.maximum(self.W_data,
                                  np.outer(data_vec, addr_decoded))

    def _data_read(self, addr_decoded: np.ndarray) -> np.ndarray:
        """Dot product then top-N_d re-encoded as significance vector.
        Thesis section 4.2: 'd neurons with maximum activations ... sorted
        and ordered' to form the output significance vector."""
        raw = self.W_data @ addr_decoded
        top = topk_indices(raw, self.N_d)
        top_sorted = top[np.argsort(raw[top])[::-1]]
        out = np.zeros(self.D, dtype=np.float64)
        for rank, idx in enumerate(top_sorted):
            out[idx] = self.alpha ** rank
        return out
