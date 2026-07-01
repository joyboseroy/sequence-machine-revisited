"""
Head-to-head comparison of all four SDM variants on the same data:
  (a) StandardSDM   — binary codes, OR weights
  (b) RankOrderSDM  — significance vectors, MAX weights
  (c) WheelSDM      — significance vectors, wheel spiking decode
  (d) RDLIFSDM      — significance vectors, RDLIF spiking decode

Tests:
  1. Single write/read round-trip (sanity check)
  2. Memory capacity curve (pairs_stored vs pairs_recalled correctly)
  3. Equivalence check: do (b), (c), (d) give the same outputs?
     The thesis claims wheel model can be made equivalent to (b).
     RDLIF is expected to diverge slightly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sdm_library import (
    StandardSDM, RankOrderSDM, WheelSDM, RDLIFSDM,
    make_binary_nofm, make_significance_vectors, normalised_dot_product,
)

# ---- shared parameters (thesis Ch4 section 4.3.5 defaults, scaled down
#      so RDLIF numerical simulation runs fast enough for a demo) ----
D, N_d = 64, 6        # d-of-D code (thesis default 11-of-256; scaled down)
W, N_w = 256, 8       # w-of-W address decoder (thesis: 16-of-4096; scaled)
ALPHA  = 0.99         # significance ratio (thesis default)
SEED   = 42
N_a    = 20     # a: address decoder weight sparsity (a-of-A, separate from N_d)
RNG    = np.random.default_rng(SEED)


def make_sdms():
    return {
        "(a) Standard":   StandardSDM( D, N_i=N_d, N_a=N_a, W=W, N_w=N_w, N_d=N_d, alpha=ALPHA, seed=SEED),
        "(b) RankOrder":  RankOrderSDM(D, N_i=N_d, N_a=N_a, W=W, N_w=N_w, N_d=N_d, alpha=ALPHA, seed=SEED),
        "(c) Wheel":      WheelSDM(    D, N_i=N_d, N_a=N_a, W=W, N_w=N_w, N_d=N_d, alpha=ALPHA, seed=SEED,
                                        slope=1.0, threshold=1000.0),
        "(d) RDLIF":      RDLIFSDM(    D, N_i=N_d, N_a=N_a, W=W, N_w=N_w, N_d=N_d, alpha=ALPHA, seed=SEED,
                                        tau_r=0.5, tau_a=2.0,
                                        threshold=0.05, t_window=15.0),
    }


# ---- 1. single round-trip sanity check ----
def test_single_roundtrip():
    print("=" * 60)
    print("1. Single write/read round-trip")
    print("=" * 60)
    bin_codes = make_binary_nofm(2, D, N_d, RNG)
    sig_codes = make_significance_vectors(2, D, N_d, ALPHA, RNG)

    sdms = make_sdms()
    for name, sdm in sdms.items():
        addr = bin_codes[0] if "Standard" in name else sig_codes[0]
        data = bin_codes[1] if "Standard" in name else sig_codes[1]
        sdm.write(addr, data)
        recalled = sdm.read(addr)
        sim = normalised_dot_product(recalled, data)
        print(f"  {name:20s}  similarity={sim:.4f}")
    print()


# ---- 2. memory capacity curve ----
def test_capacity(n_pairs_list=None):
    if n_pairs_list is None:
        n_pairs_list = [5, 10, 20, 40, 80, 120, 160]
    print("=" * 60)
    print("2. Memory capacity (pairs stored vs pairs recalled, sim >= 0.9)")
    print("=" * 60)
    print(f"{'n_pairs':>8}", end="")
    names = ["(a) Standard", "(b) RankOrder", "(c) Wheel", "(d) RDLIF"]
    for n in names:
        print(f"  {n:>15}", end="")
    print()

    for n in n_pairs_list:
        bin_codes_addr = make_binary_nofm(n, D, N_d, RNG)
        bin_codes_data = make_binary_nofm(n, D, N_d, RNG)
        sig_codes_addr = make_significance_vectors(n, D, N_d, ALPHA, RNG)
        sig_codes_data = make_significance_vectors(n, D, N_d, ALPHA, RNG)

        sdms = make_sdms()
        print(f"{n:>8}", end="")
        for name, sdm in sdms.items():
            is_standard = "Standard" in name
            addr_c = bin_codes_addr if is_standard else sig_codes_addr
            data_c = bin_codes_data if is_standard else sig_codes_data
            result = sdm.capacity_test(addr_c, data_c, similarity_threshold=0.9)
            print(f"  {result['correct']:>6}/{n:<6}", end="")
        print()
    print()


# ---- 3. equivalence: do (b), (c), (d) give the same output? ----
def test_equivalence(n_pairs=5):
    print("=" * 60)
    print(f"3. Equivalence check: (b) vs (c) vs (d) on {n_pairs} pairs")
    print("   (thesis claim: wheel model should match abstract model exactly)")
    print("=" * 60)
    sig_addr = make_significance_vectors(n_pairs, D, N_d, ALPHA, RNG)
    sig_data = make_significance_vectors(n_pairs, D, N_d, ALPHA, RNG)

    ro  = RankOrderSDM(D, N_i=N_d, N_a=N_a, W=W, N_w=N_w, N_d=N_d, alpha=ALPHA, seed=SEED)
    wh  = WheelSDM(    D, N_i=N_d, N_a=N_a, W=W, N_w=N_w, N_d=N_d, alpha=ALPHA, seed=SEED, threshold=1000.0)
    rd  = RDLIFSDM(    D, N_i=N_d, N_a=N_a, W=W, N_w=N_w, N_d=N_d, alpha=ALPHA, seed=SEED, tau_a=2.0,
                        threshold=0.05, t_window=15.0)

    for i in range(n_pairs):
        ro.write(sig_addr[i], sig_data[i])
        wh.write(sig_addr[i], sig_data[i])
        rd.write(sig_addr[i], sig_data[i])

    print(f"  {'pair':>4}  {'RO-vs-Wheel':>12}  {'RO-vs-RDLIF':>12}  {'Wheel-vs-RDLIF':>15}")
    for i in range(n_pairs):
        out_ro = ro.read(sig_addr[i])
        out_wh = wh.read(sig_addr[i])
        out_rd = rd.read(sig_addr[i])
        print(f"  {i:>4}  "
              f"{normalised_dot_product(out_ro, out_wh):>12.4f}  "
              f"{normalised_dot_product(out_ro, out_rd):>12.4f}  "
              f"{normalised_dot_product(out_wh, out_rd):>15.4f}")
    print()
    print("  RO-vs-Wheel ≈ 1.0 means wheel model exactly matches abstract model.")
    print("  RO-vs-RDLIF may diverge due to timing-sensitivity of RDLIF neurons.")
    print("  (thesis Chapter 6 p.125 documents this divergence as a known limitation)")


if __name__ == "__main__":
    test_single_roundtrip()
    test_capacity()
    test_equivalence()
