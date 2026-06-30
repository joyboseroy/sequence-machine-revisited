"""
Reproduce Figures 4-5 of the paper: CMM memory capacity as a function of
number of stored address-data pairs, comparing non-spiking and spiking
(analytic-LIF) decode, for A=D=256, i=d=11 (11-of-256 codes) as specified
in Section IV.B.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sdm_nengo_2021 import random_nofm_codes, NonSpikingCMM, SpikingCMM


def measure_capacity(cmm_class, A, D, N, pair_counts, seed=0):
    """For each count in pair_counts, build a FRESH memory, write that many
    random pairs, then count how many of THOSE pairs are recalled exactly
    correctly. This matches the paper's methodology: 'vary the number of
    address-data pairs stored... and plot the number of pairs recalled
    correctly against the number of pairs written.'"""
    rng = np.random.default_rng(seed)
    results = []
    for q in pair_counts:
        addresses = random_nofm_codes(q, A, N, rng)
        datas = random_nofm_codes(q, D, N, rng)
        cmm = cmm_class(A=A, D=D, d=N)
        for k in range(q):
            cmm.write(addresses[k], datas[k])
        correct = 0
        for k in range(q):
            recalled = cmm.read(addresses[k])
            if np.array_equal(recalled, datas[k]):
                correct += 1
        results.append(correct)
    return results


def main():
    A = D = 256
    N = 11
    pair_counts = list(range(25, 525, 25))

    print(f"=== CMM memory capacity, A=D={A}, {N}-of-{A} codes ===")
    print(f"{'pairs_stored':>13} {'non_spiking_correct':>20} {'spiking_LIF_correct':>20}")
    non_spiking = measure_capacity(NonSpikingCMM, A, D, N, pair_counts, seed=1)
    spiking = measure_capacity(SpikingCMM, A, D, N, pair_counts, seed=1)
    for q, ns, sp in zip(pair_counts, non_spiking, spiking):
        print(f"{q:>13} {ns:>20} {sp:>20}")

    peak_ns = pair_counts[int(np.argmax(non_spiking))]
    peak_sp = pair_counts[int(np.argmax(spiking))]
    print(f"\nPeak capacity: non-spiking at q={peak_ns}, spiking at q={peak_sp}")
    print("(paper reports perfect recall up to ~300, plateau 300-350, "
          "gradual decline beyond 350)")


if __name__ == "__main__":
    main()
