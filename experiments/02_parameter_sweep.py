"""
Small parameter sweep to better match the optimised parameters the 2005
paper reports (lambda=0.2 for context layer, x=1.0 for combined, K=2*N).
We don't have their exact sweep, so we do a coarse grid search of our own
and report the best-performing setting per model, on a fixed test sequence.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
from sequence_machine_2005 import (
    SequenceMachine2005, ShiftRegisterContext, ContextLayerModel, CombinedContext
)


def random_sequence(alphabet_size, length, seed):
    rng = random.Random(seed)
    return [rng.randrange(alphabet_size) for _ in range(length)]


def count_errors(machine, seq):
    machine.run_sequence(seq)
    machine.reset()
    errors = 0
    for i, sym in enumerate(seq):
        pred = machine.step(sym)
        if i + 1 < len(seq) and pred != seq[i + 1]:
            errors += 1
    return errors


def main():
    alphabet_size = 15
    M, N = 256, 11
    addr_dim = 2048
    test_lengths = [30, 60, 90]
    seqs = [random_sequence(alphabet_size, L, seed=L) for L in test_lengths]

    print("=== context layer: sweep lambda ===")
    for lam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        total_err = 0
        for seq in seqs:
            ctx = ContextLayerModel(M=M, lam=lam, seed=2)
            m = SequenceMachine2005(alphabet_size, ctx, input_M=M, input_N=N,
                                     addr_dim=addr_dim, addr_N=N, seed=2)
            total_err += count_errors(m, seq)
        print(f"  lambda={lam:.1f}  total_errors={total_err}")

    print("=== combined model: sweep x, K ===")
    for x in [0.3, 0.5, 0.7, 1.0]:
        for K in [22, 44, 88, 176]:
            total_err = 0
            for seq in seqs:
                ctx = CombinedContext(ctx_dim=512, input_dim=M, K=K, x=x, seed=3)
                m = SequenceMachine2005(alphabet_size, ctx, input_M=M, input_N=N,
                                         addr_dim=addr_dim, addr_N=N, seed=3)
                total_err += count_errors(m, seq)
            print(f"  x={x:.1f} K={K:3d}  total_errors={total_err}")


if __name__ == "__main__":
    main()
