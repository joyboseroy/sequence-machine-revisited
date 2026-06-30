"""
Long-distractor associative recall benchmark, v2: SDM vs a proper RAG /
vector-database baseline (replacing the dense-MLP strawman from v1).

Protocol: store a single key->value association, then write D unrelated
distractor associations, then probe with the original cue and check
whether the target is still recoverable. This tests long-term associative
recall under interference -- the concrete version of "ABCDEF -> XYZ after
10,000 distractions."

Three memories are compared:

1. SDM (fixed-size, content-addressable, max-Hebbian write) -- same as v1.
2. UnboundedVectorDB -- the standard naive RAG setup: store everything,
   retrieve by nearest-neighbour cosine similarity, no capacity limit.
   This is what most "RAG for agent memory" systems look like in
   practice. Expect this to succeed at every distractor count by
   construction, UNLESS embedding collisions cause retrieval errors --
   which is the actually interesting failure mode to look for, not
   "running out of room."
3. CappedVectorDB -- the fair, apples-to-apples comparison: a vector
   store with the SAME memory budget (number of slots) as the SDM's
   address-decoder dimensionality, using FIFO eviction once full. This is
   what happens to a RAG memory once it has a real-world storage/context
   budget, which it always eventually does.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
from sequence_machine_2005 import make_rank_order_codes, decode, KanervaSDM
from rag_baselines import UnboundedVectorDB, CappedVectorDB


def sdm_recall_test(vocab_size, num_distractors, addr_dim=2048, addr_N=11,
                     M=256, N=11, seed=0):
    codes = make_rank_order_codes(vocab_size, M, N, seed=seed)
    sdm = KanervaSDM(context_dim=M, addr_dim=addr_dim, input_dim=M, addr_N=addr_N, seed=seed)

    rng = random.Random(seed)
    cue, target = 0, 1
    sdm.write(codes[cue], codes[target])

    for _ in range(num_distractors):
        a = rng.randrange(vocab_size)
        while a == cue:
            a = rng.randrange(vocab_size)
        b = rng.randrange(vocab_size)
        sdm.write(codes[a], codes[b])

    recalled_vec = sdm.read(codes[cue])
    recalled = decode(recalled_vec, codes)
    return recalled == target


def vector_db_recall_test(db_class, vocab_size, num_distractors, M=256, N=11,
                           capacity=None, seed=0):
    codes_t = make_rank_order_codes(vocab_size, M, N, seed=seed)
    codes = codes_t.numpy()  # rag_baselines.py works in numpy, not torch
    db = db_class(embed_dim=M, seed=seed) if capacity is None else \
        db_class(embed_dim=M, capacity=capacity, seed=seed)

    rng = random.Random(seed)
    cue, target = 0, 1
    db.write(codes[cue], target)

    for _ in range(num_distractors):
        a = rng.randrange(vocab_size)
        while a == cue:
            a = rng.randrange(vocab_size)
        b = rng.randrange(vocab_size)
        db.write(codes[a], b)

    recalled = db.read(codes[cue])
    return recalled == target


def main():
    vocab_size = 50
    M, N = 256, 11
    addr_dim = 2048  # SDM's memory budget -> also CappedVectorDB's budget
    distractor_counts = [0, 10, 50, 200, 1000, 5000, 20000]

    print(f"=== SDM (fixed budget: {addr_dim} address-decoder slots) ===")
    for D in distractor_counts:
        ok = sdm_recall_test(vocab_size, D, addr_dim=addr_dim, M=M, N=N)
        print(f"  distractors={D:6d}  cue recalled correctly: {ok}")

    print("\n=== UnboundedVectorDB (naive RAG: no capacity limit) ===")
    for D in distractor_counts:
        ok = vector_db_recall_test(UnboundedVectorDB, vocab_size, D, M=M, N=N)
        print(f"  distractors={D:6d}  cue recalled correctly: {ok}")

    print(f"\n=== CappedVectorDB (RAG with SAME memory budget as SDM: "
          f"{addr_dim} slots, FIFO eviction) ===")
    for D in distractor_counts:
        ok = vector_db_recall_test(CappedVectorDB, vocab_size, D, M=M, N=N,
                                    capacity=addr_dim)
        print(f"  distractors={D:6d}  cue recalled correctly: {ok}")


if __name__ == "__main__":
    main()
