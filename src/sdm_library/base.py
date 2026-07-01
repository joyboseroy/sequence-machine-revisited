"""
sdm_library/base.py

Shared encoding utilities and abstract base class for all four SDM variants.

All four SDM variants share:
  - The same two-layer architecture (address decoder + data store)
  - The same code generation / similarity measurement interface
  - Different only in HOW they encode symbols and HOW they compute
    address-decoder outputs and data-store reads/writes.

Thesis parameters (Chapter 4, section 4.3.5 defaults):
  d-of-D = 11-of-256  (address and data code)
  w-of-W = 16-of-4096 (address decoder output code)
  alpha   = 0.99       (significance/desensitisation ratio)
  similarity threshold = 0.9 for perfect-match capacity
"""

from abc import ABC, abstractmethod
import numpy as np


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def make_binary_nofm(num_symbols: int, M: int, N: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Generate num_symbols random binary N-of-M codes.
    Each row has exactly N ones and (M-N) zeros.
    Used for: (a) StandardSDM symbol encoding."""
    codes = np.zeros((num_symbols, M), dtype=np.float64)
    for i in range(num_symbols):
        idx = rng.choice(M, size=N, replace=False)
        codes[i, idx] = 1.0
    return codes


def make_significance_vectors(num_symbols: int, M: int, N: int,
                               alpha: float,
                               rng: np.random.Generator) -> np.ndarray:
    """Generate num_symbols rank-ordered N-of-M significance vectors.

    Per thesis Chapter 4, section 4.1.2: each active component is assigned
    a significance of alpha^rank where rank 0 is the first to fire (highest
    significance = 1.0), rank 1 has significance alpha, rank 2 has alpha^2,
    etc. All other (inactive) components are 0.

    Example from thesis (3-of-5, neurons 3,2,4 fire in order):
        vector = [0, alpha, 1, alpha^2, 0]
    """
    codes = np.zeros((num_symbols, M), dtype=np.float64)
    for i in range(num_symbols):
        active = rng.choice(M, size=N, replace=False)
        order = rng.permutation(N)          # random firing order
        for rank, j in enumerate(order):
            codes[i, active[j]] = alpha ** rank
    return codes


def normalised_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity (normalised dot product), per thesis eq. 4.1.
    Returns 1.0 if identical, 0.0 if orthogonal, in [0,1] for non-negative
    significance vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def topk_indices(vec: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k largest components (L-max thresholding,
    thesis eq. 3.11). Ties broken by index order."""
    return np.argpartition(vec, -k)[-k:]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SDMBase(ABC):
    """Abstract base for all four SDM variants.

    Subclasses implement _encode, _addr_decode, _write_pair, _read_addr.
    The public interface (write / read / similarity / capacity_test) is
    shared across all four variants so they can be used interchangeably.
    """

    def __init__(self, D: int, N_d: int, W: int, N_w: int,
                 alpha: float = 0.99, seed: int = 0):
        """
        D     : dimensionality of address/data vectors (e.g. 256)
        N_d   : number of active components in each d-of-D code (e.g. 11)
        W     : number of address decoder neurons (e.g. 4096)
        N_w   : number of active components in address decoder output (e.g. 16)
        alpha : significance ratio / desensitisation factor (e.g. 0.99)
        seed  : random seed for reproducibility
        """
        self.D = D
        self.N_d = N_d
        self.W = W
        self.N_w = N_w
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self._init_weights()

    @abstractmethod
    def _init_weights(self):
        """Initialise address decoder (fixed) and data store (zeros)."""

    @abstractmethod
    def _addr_forward(self, address_vec: np.ndarray) -> np.ndarray:
        """Pass address_vec through address decoder -> return w-of-W vector."""

    @abstractmethod
    def _data_write(self, addr_decoded: np.ndarray, data_vec: np.ndarray):
        """Update data store weights associating addr_decoded with data_vec."""

    @abstractmethod
    def _data_read(self, addr_decoded: np.ndarray) -> np.ndarray:
        """Read data store using addr_decoded -> return raw D-dim output."""

    # -- public interface ----------------------------------------------------

    def write(self, address_vec: np.ndarray, data_vec: np.ndarray):
        """Associate address_vec -> data_vec in the memory."""
        addr_decoded = self._addr_forward(address_vec)
        self._data_write(addr_decoded, data_vec)

    def read(self, address_vec: np.ndarray) -> np.ndarray:
        """Recall the data vector associated with address_vec.
        Returns the raw D-dim output vector (before symbol decoding)."""
        addr_decoded = self._addr_forward(address_vec)
        return self._data_read(addr_decoded)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return normalised_dot_product(a, b)

    def reset(self):
        """Clear the data store (address decoder weights are fixed)."""
        self._init_weights()
        # re-initialise only data store, not address decoder
        # subclasses may override if needed

    def memory_occupancy(self) -> float:
        """Fraction of data store weight matrix entries that are non-zero.
        Per thesis section 4.3.5 definition."""
        total = self.W_data.size
        nonzero = np.count_nonzero(self.W_data)
        return nonzero / total if total > 0 else 0.0

    def capacity_test(self, address_codes: np.ndarray,
                      data_codes: np.ndarray,
                      similarity_threshold: float = 0.9) -> dict:
        """Write N pairs then read all back. Returns:
          correct      : count recalled above similarity_threshold
          avg_sim      : mean similarity across all pairs
          occupancy    : memory occupancy at end
        Per thesis section 4.3.3 experimental procedure."""
        n = len(address_codes)
        for i in range(n):
            self.write(address_codes[i], data_codes[i])
        correct = 0
        sims = []
        for i in range(n):
            recalled = self.read(address_codes[i])
            sim = self.similarity(recalled, data_codes[i])
            sims.append(sim)
            if sim >= similarity_threshold:
                correct += 1
        return {
            "n": n,
            "correct": correct,
            "avg_sim": float(np.mean(sims)),
            "occupancy": self.memory_occupancy(),
        }
