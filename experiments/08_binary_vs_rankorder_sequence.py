"""
Experiment 08: Standard (binary) SDM vs Rank-Order SDM — clean comparison.

Two tests:

TEST A — Isolated SDM capacity (no context model).
  Write n random (address, data) pairs directly into each SDM, then read
  them back. This isolates the encoding question cleanly: does rank-order
  give better capacity than binary, holding memory size equal?
  Matches the thesis Chapter 4 experimental procedure exactly.

TEST B — Sequence prediction with forced context collisions.
  Construct sequences with a SHARED common subsequence (the thesis's own
  repeated-subsequence test, Table I / section IX.B of IJCNN paper).
  Both machines must use the SAME context to predict two DIFFERENT next
  symbols — this is the interference condition where encoding quality
  actually matters for discrimination.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
import numpy as np
from sdm_library import (
    StandardSDM, RankOrderSDM,
    make_binary_nofm, make_significance_vectors,
    normalised_dot_product, topk_indices,
)

D       = 256
N_i     = 11   # i: input address code sparsity (i-of-A)
N_a     = 20   # a: address decoder weight sparsity (a-of-A) — SEPARATE from N_i
N_d     = 11   # d: data code sparsity (d-of-D)
W, N_w  = 4096, 16  # thesis Ch4 default: 16-of-4096
ALPHA   = 0.99
SEED    = 3
RNG     = np.random.default_rng(SEED)


# ============================================================
# TEST A: isolated SDM capacity
# ============================================================

def test_a_capacity():
    print("=" * 62)
    print("TEST A: Isolated SDM capacity (thesis Ch4 protocol)")
    print(f"  i-of-A={N_i}-of-{D}, a-of-A={N_a}-of-{D}, w-of-W={N_w}-of-{W}, d-of-D={N_d}-of-{D}, alpha={ALPHA}")
    print("=" * 62)
    print(f"{'n_pairs':>8}  {'standard correct':>17}  "
          f"{'rankorder correct':>18}  {'std_acc':>8}  {'ro_acc':>7}")

    for n in [50, 100, 200, 400, 700, 1000, 1500, 2000]:
        bin_addr = make_binary_nofm(n, D, N_i, RNG)
        bin_data = make_binary_nofm(n, D, N_d, RNG)
        sig_addr = make_significance_vectors(n, D, N_i, ALPHA, RNG)
        sig_data = make_significance_vectors(n, D, N_d, ALPHA, RNG)

        std = StandardSDM( D, N_i=N_i, N_a=N_a, W=W, N_w=N_w, N_d=N_d, alpha=ALPHA, seed=SEED)
        ro  = RankOrderSDM(D, N_i=N_i, N_a=N_a, W=W, N_w=N_w, N_d=N_d, alpha=ALPHA, seed=SEED)

        r_std = std.capacity_test(bin_addr, bin_data, similarity_threshold=0.9)
        r_ro  = ro.capacity_test( sig_addr, sig_data, similarity_threshold=0.9)

        print(f"{n:>8}  {r_std['correct']:>8}/{n:<8}  "
              f"{r_ro['correct']:>9}/{n:<8}  "
              f"{r_std['correct']/n:>8.1%}  {r_ro['correct']/n:>7.1%}")
    print()


# ============================================================
# TEST B: sequence prediction with forced context collisions
# ============================================================

CTX_DIM = 512
CTX_K   = 22
LAMBDA  = 0.5
ALPHABET = 12   # small alphabet -> more forced repeats

rng_ctx = np.random.default_rng(SEED + 1)
P1 = rng_ctx.random((CTX_DIM, CTX_DIM))
P2 = rng_ctx.random((CTX_DIM, D))
P_ctx = rng_ctx.random((D, CTX_DIM))    # ctx -> SDM address

def update_ctx(ctx, input_vec):
    s_old = P1 @ ctx;  s_old /= (s_old.sum() + 1e-12)
    s_new = P2 @ input_vec; s_new /= (s_new.sum() + 1e-12)
    combined = LAMBDA * s_old + (1 - LAMBDA) * s_new
    top = topk_indices(combined, CTX_K)
    top_s = top[np.argsort(combined[top])[::-1]]
    out = np.zeros(CTX_DIM)
    for r, idx in enumerate(top_s): out[idx] = ALPHA ** r
    return out

def project_ctx_binary(ctx):
    raw = P_ctx @ ctx
    out = np.zeros(D)
    out[topk_indices(raw, N_d)] = 1.0
    return out

def project_ctx_rank(ctx):
    raw = P_ctx @ ctx
    top = topk_indices(raw, N_d)
    top_s = top[np.argsort(raw[top])[::-1]]
    out = np.zeros(D)
    for r, idx in enumerate(top_s): out[idx] = ALPHA ** r
    return out

def decode_binary(vec, cb): return int(np.argmax(cb @ vec))
def decode_rank(vec, cb):
    return int(np.argmax([normalised_dot_product(vec, cb[s])
                          for s in range(len(cb))]))

def make_collision_seq(common_len=4, branch_len=5, n_branches=4):
    """seq = [branch0][COMMON][branch1][COMMON][branch2]...
    The COMMON subsequence appears before different continuations,
    forcing the SDM to use context to disambiguate — the thesis's
    own repeated-subsequence test (IJCNN paper Table I)."""
    common = list(range(common_len))           # symbols 0..common_len-1
    used = set(common)
    branches = []
    sym = common_len
    for _ in range(n_branches):
        b = []
        for _ in range(branch_len):
            while sym in used: sym += 1
            b.append(sym % ALPHABET); sym += 1
        branches.append(b)
    seq = []
    for b in branches:
        seq += b + common
    return seq

def run_collision(sdm, codebook, decode_fn, proj_fn, seq):
    ctx = np.zeros(CTX_DIM)
    for sym in seq:
        sdm.write(proj_fn(ctx), codebook[sym])
        ctx = update_ctx(ctx, codebook[sym])
    ctx = np.zeros(CTX_DIM)
    errors = 0
    for i, sym in enumerate(seq):
        sdm.write(proj_fn(ctx), codebook[sym])
        ctx = update_ctx(ctx, codebook[sym])
        if i + 1 < len(seq):
            pred = decode_fn(sdm.read(proj_fn(ctx)), codebook)
            if pred != seq[i + 1]:
                errors += 1
    return errors, len(seq) - 1

def test_b_collisions():
    print("=" * 62)
    print("TEST B: Repeated-subsequence (context collision) test")
    print("  Same common subsequence precedes different continuations.")
    print("  SDM must use context to discriminate — encoding matters.")
    print("=" * 62)

    bin_cb = make_binary_nofm(ALPHABET, D, N_i, RNG)
    sig_cb = make_significance_vectors(ALPHABET, D, N_i, ALPHA, RNG)

    print(f"{'common_len':>11}  {'branches':>8}  "
          f"{'std_errors':>11}  {'ro_errors':>10}  "
          f"{'std_acc':>8}  {'ro_acc':>7}")

    for common_len in [2, 3, 4, 5, 6]:
        for n_branches in [3, 5]:
            seq = make_collision_seq(common_len=common_len,
                                     branch_len=5,
                                     n_branches=n_branches)
            std = StandardSDM( D, N_i=N_i, N_a=N_a, W=W, N_w=N_w, N_d=N_d, alpha=ALPHA, seed=SEED)
            ro  = RankOrderSDM(D, N_i=N_i, N_a=N_a, W=W, N_w=N_w, N_d=N_d, alpha=ALPHA, seed=SEED)

            std_err, total = run_collision(
                std, bin_cb, decode_binary, project_ctx_binary, seq)
            ro_err, _      = run_collision(
                ro,  sig_cb, decode_rank,   project_ctx_rank,   seq)

            print(f"{common_len:>11}  {n_branches:>8}  "
                  f"{std_err:>11}  {ro_err:>10}  "
                  f"{1-std_err/total:>8.1%}  {1-ro_err/total:>7.1%}")
    print()
    print("If rank-order encoding helps: ro_acc >= std_acc especially")
    print("at longer common subsequences where context overlap is hardest.")


if __name__ == "__main__":
    test_a_capacity()
    test_b_collisions()
