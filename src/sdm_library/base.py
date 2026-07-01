"""
sdm_library/base.py

Shared encoding utilities and abstract base class for all four SDM variants.

The thesis (Chapter 3, Figure 3.2, p.78) defines FOUR distinct N-of-M
parameters — not two. Getting this right is the key architectural point:

  i-of-A  : input address code (e.g. 11-of-256)
             Each input symbol has exactly i active bits out of A total.
  a-of-A  : address decoder weight code (e.g. 20-of-256)
             Each of the W address decoder neurons has exactly a non-zero
             weights connecting to the A inputs. a is a SEPARATE parameter
             from i — it controls how many of the W decoders fire in response
             to a given input. Chosen so approximately w out of W fire.
  w-of-W  : address decoder output code (e.g. 16-of-4096)
             The top-w firing address decoders form the high-dimensional
             address word passed to the data store.
  d-of-D  : data code (e.g. 11-of-256)
             Each data symbol has exactly d active bits out of D total.
             Often i=d and A=D, but not required by the thesis.

Thesis reference: Chapter 3 section 3.3.5, Figure 3.2, eq. 3.8-3.11.
Chapter 4 section 4.3.5 default parameters:
  i=d=11, A=D=256, w=16, W=4096, alpha=0.99
  (a is not stated explicitly but typically a > i, e.g. a=20 per Ajwani
   et al. 2021 who use the same Furber SDM architecture)
"""

from abc import ABC, abstractmethod
import numpy as np


# ---------------------------------------------------------------------------
# Encoding utilities
# ---------------------------------------------------------------------------

def make_binary_nofm(num_symbols: int, M: int, N: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Generate num_symbols random binary N-of-M codes (exactly N of M
    bits set to 1). Used for i-of-A input codes in StandardSDM."""
    codes = np.zeros((num_symbols, M), dtype=np.float64)
    for i in range(num_symbols):
        idx = rng.choice(M, size=N, replace=False)
        codes[i, idx] = 1.0
    return codes


def make_significance_vectors(num_symbols: int, M: int, N: int,
                               alpha: float,
                               rng: np.random.Generator) -> np.ndarray:
    """Generate num_symbols rank-ordered N-of-M significance vectors.

    Per thesis Chapter 4 section 4.1.2: each of the N active components
    gets weight alpha^rank where rank 0 = first to fire (significance 1.0),
    rank 1 gets alpha, rank 2 gets alpha^2, etc. All inactive = 0.

    Example (thesis p.84): 3-of-5, neurons 3,2,4 fire in order:
        vector = [0, alpha, 1, alpha^2, 0]
    """
    codes = np.zeros((num_symbols, M), dtype=np.float64)
    for i in range(num_symbols):
        active = rng.choice(M, size=N, replace=False)
        order = rng.permutation(N)
        for rank, j in enumerate(order):
            codes[i, active[j]] = alpha ** rank
    return codes


def normalised_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity (normalised dot product), thesis eq. 4.1."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def topk_indices(vec: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k largest components (L-max, thesis eq. 3.11).
    Ties broken by index order."""
    if k >= len(vec):
        return np.arange(len(vec))
    return np.argpartition(vec, -k)[-k:]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SDMBase(ABC):
    """Abstract base for all four SDM variants.

    Parameters follow thesis Figure 3.2 exactly:
      A, N_i : dimensionality and sparsity of input address code (i-of-A)
      A, N_a : dimensionality and sparsity of address decoder weights (a-of-A)
               NOTE: N_a is a SEPARATE parameter from N_i
      W, N_w : number of address decoders and output code sparsity (w-of-W)
      D, N_d : dimensionality and sparsity of data code (d-of-D)
               Often A=D and N_i=N_d but not required.

    For simplicity we assume A=D (same dimensionality for address and data)
    which is the case in all thesis experiments. N_i=N_d=11, N_a=20, W=4096,
    N_w=16 are the thesis Chapter 4 defaults.
    """

    def __init__(self,
                 D: int = 256,    # A = D (address and data dimensionality)
                 N_i: int = 11,   # i: input address code sparsity (i-of-A)
                 N_a: int = 20,   # a: address decoder weight sparsity (a-of-A)
                 W: int = 4096,   # W: number of address decoder neurons
                 N_w: int = 16,   # w: address decoder output sparsity (w-of-W)
                 N_d: int = 11,   # d: data code sparsity (d-of-D)
                 alpha: float = 0.99,
                 seed: int = 0):
        self.D   = D       # A = D
        self.N_i = N_i
        self.N_a = N_a
        self.W   = W
        self.N_w = N_w
        self.N_d = N_d
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self._init_weights()

    @abstractmethod
    def _init_weights(self):
        """Initialise address decoder (fixed) and data store (zeros)."""

    @abstractmethod
    def _addr_forward(self, address_vec: np.ndarray) -> np.ndarray:
        """Pass i-of-A address_vec through decoder -> return w-of-W vector."""

    @abstractmethod
    def _data_write(self, addr_decoded: np.ndarray, data_vec: np.ndarray):
        """Update data store weights."""

    @abstractmethod
    def _data_read(self, addr_decoded: np.ndarray) -> np.ndarray:
        """Read data store -> return raw D-dim output vector."""

    # -- public interface ----------------------------------------------------

    def write(self, address_vec: np.ndarray, data_vec: np.ndarray):
        addr_decoded = self._addr_forward(address_vec)
        self._data_write(addr_decoded, data_vec)

    def read(self, address_vec: np.ndarray) -> np.ndarray:
        addr_decoded = self._addr_forward(address_vec)
        return self._data_read(addr_decoded)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return normalised_dot_product(a, b)

    def reset(self):
        self._init_weights()

    def memory_occupancy(self) -> float:
        """Fraction of data store weight matrix entries that are non-zero.
        Per thesis section 4.3.5 definition."""
        total = self.W_data.size
        nonzero = np.count_nonzero(self.W_data)
        return nonzero / total if total > 0 else 0.0

    def capacity_test(self, address_codes: np.ndarray,
                      data_codes: np.ndarray,
                      similarity_threshold: float = 0.9) -> dict:
        """Write N pairs then read all back. Returns correct count, avg
        similarity, and memory occupancy. Per thesis section 4.3.3."""
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
