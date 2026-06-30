"""
Reproduce Figures 6-9 of the paper: SDM memory capacity as a function of
number of stored address-data pairs, for address decoder sizes W in
{256, 512, 1024}, comparing non-spiking and spiking (analytic-LIF) decode.
Parameters per Section IV.B: A=256, i=11 (address code), w=16 (address
decoder output code, w-of-W), D=256, d=11 (data code).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sdm_nengo_2021 import random_nofm_codes, NonSpikingSDM, SpikingSDM


def measure_sdm_capacity(sdm_class, A, W, w, D, N, pair_counts, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for q in pair_counts:
        addresses = random_nofm_codes(q, A, N, rng)
        datas = random_nofm_codes(q, D, N, rng)
        sdm = sdm_class(A=A, W=W, w=w, D=D, d=N, rng=rng)
        for k in range(q):
            sdm.write(addresses[k], datas[k])
        correct = 0
        for k in range(q):
            recalled = sdm.read(addresses[k])
            if np.array_equal(recalled, datas[k]):
                correct += 1
        results.append(correct)
    return results


def main():
    A = D = 256
    N = 11      # address/data code N-of-256
    w = 16      # address decoder output code, w-of-W

    for W in [256, 512, 1024]:
        # scale the test range with W, since capacity grows with address
        # decoder size (per paper's Fig 9 observation)
        max_q = int(W * 1.8)
        step = max(25, max_q // 20)
        pair_counts = list(range(25, max_q, step))

        print(f"\n=== SDM memory capacity, W={W} (address decoder size, "
              f"{w}-of-{W} codes) ===")
        print(f"{'pairs_stored':>13} {'non_spiking_correct':>20} {'spiking_LIF_correct':>20}")
        non_spiking = measure_sdm_capacity(NonSpikingSDM, A, W, w, D, N, pair_counts, seed=2)
        spiking = measure_sdm_capacity(SpikingSDM, A, W, w, D, N, pair_counts, seed=2)
        for q, ns, sp in zip(pair_counts, non_spiking, spiking):
            print(f"{q:>13} {ns:>20} {sp:>20}")

        peak_ns = pair_counts[int(np.argmax(non_spiking))]
        peak_sp = pair_counts[int(np.argmax(spiking))]
        print(f"Peak capacity (W={W}): non-spiking at q={peak_ns}, spiking at q={peak_sp}")

    print("\n(paper Fig 9: peak capacity grows with address decoder size W, "
          "shape of curve similar across W)")


if __name__ == "__main__":
    main()
