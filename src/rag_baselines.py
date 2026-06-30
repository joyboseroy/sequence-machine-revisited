"""
Proper RAG / vector-database baselines for the long-distractor associative
recall benchmark, replacing the dense-MLP strawman from the first version
of experiment 04.

The dense MLP baseline was misleading: it conflates two different failure
modes (gradient-training capacity AND a tiny fixed-epoch budget) and isn't
what a real "RAG for agent memory" system looks like. A real RAG memory is
a vector store: embed the cue, store (embedding, value) pairs, and on query
retrieve via nearest-neighbour similarity search. This module implements
two honest versions of that:

1. UnboundedVectorDB: the standard, naive RAG setup. No capacity limit,
   exact storage, cosine-similarity nearest-neighbour retrieval. This is
   what most people mean by "RAG memory" in practice (e.g. a Pinecone/
   Chroma/FAISS index that just keeps growing). By construction this
   should retrieve perfectly under exact-match conditions regardless of
   distractor count, UNLESS embeddings collide (two different cues end up
   closer to each other than to their own stored value) — which is the
   actual failure mode worth measuring, not "running out of capacity."

2. CappedVectorDB: a capacity-constrained vector store with the SAME
   memory budget (number of stored slots) as the SDM's address-decoder
   dimensionality, using FIFO eviction once full. This is the fairer,
   apples-to-apples comparison: what happens to a RAG memory when it has
   a fixed budget, the way any real deployed system eventually does
   (context window limits, storage cost limits, retention policies)?

Both use random projections as a stand-in for a learned embedding model
(e.g. a sentence-transformer), since the comparison here is about the
retrieval/storage MECHANISM (exact KNN over a growing/bounded store) vs.
the SDM's fixed-size, content-addressable, max-Hebbian-write mechanism --
not about embedding quality, which is an orthogonal concern.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np


def random_nofm_codes(num_items: int, M: int, N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate num_items random binary N-of-M codes (exactly N of M bits
    set to 1)."""
    codes = np.zeros((num_items, M), dtype=np.float64)
    for i in range(num_items):
        active = rng.choice(M, size=N, replace=False)
        codes[i, active] = 1.0
    return codes


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class UnboundedVectorDB:
    """Naive RAG memory: store every (embedding, value) pair, retrieve by
    nearest-neighbour cosine similarity. No capacity limit -- this is the
    standard way most RAG/agent-memory systems are described in practice."""

    def __init__(self, embed_dim: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.proj = self.rng.standard_normal((embed_dim, embed_dim))
        self.keys = []     # embeddings
        self.values = []   # associated values (symbol ids)

    def _embed(self, code: np.ndarray) -> np.ndarray:
        return self.proj @ code

    def write(self, cue_code: np.ndarray, value):
        self.keys.append(self._embed(cue_code))
        self.values.append(value)

    def read(self, cue_code: np.ndarray):
        if not self.keys:
            return None
        q = self._embed(cue_code)
        sims = [cosine_sim(q, k) for k in self.keys]
        best = int(np.argmax(sims))
        return self.values[best]


class CappedVectorDB(UnboundedVectorDB):
    """Same retrieval mechanism, but capped at `capacity` stored items with
    FIFO eviction -- the apples-to-apples comparison against a fixed-size
    SDM. Once full, writing a new item evicts the oldest."""

    def __init__(self, embed_dim: int, capacity: int, seed: int = 0):
        super().__init__(embed_dim, seed)
        self.capacity = capacity

    def write(self, cue_code: np.ndarray, value):
        if len(self.keys) >= self.capacity:
            self.keys.pop(0)
            self.values.pop(0)
        super().write(cue_code, value)
