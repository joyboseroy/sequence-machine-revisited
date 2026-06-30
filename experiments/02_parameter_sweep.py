"""
Parameter sweep over Lambda for the thesis-faithful CombinedContext model
(thesis eq 5.5), plus a check of the thesis's own "principled" choice of
the modulation factor: lambda = alpha^(N+1) (eq. on p.105), converted to
the bounded Lambda form via Lambda = lambda/(1+lambda).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
from sequence_machine_2005 import SequenceMachine2005, CombinedContext


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
    alpha = 0.9
    test_lengths = [30, 60, 90]
    seqs = [random_sequence(alphabet_size, L, seed=L) for L in test_lengths]

    print("=== combined model (thesis eq 5.5): sweep Lambda ===")
    for Lambda in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        total_err = 0
        for seq in seqs:
            ctx = CombinedContext(ctx_dim=512, input_dim=M, m=22, Lambda=Lambda,
                                   alpha=alpha, seed=3)
            mach = SequenceMachine2005(alphabet_size, ctx, input_M=M, input_N=N,
                                        addr_dim=addr_dim, addr_N=N, seed=3)
            total_err += count_errors(mach, seq)
        print(f"  Lambda={Lambda:.1f}  total_errors={total_err}")

    print()
    print("=== thesis's own 'principled' choice: lambda = alpha^(N+1) ===")
    principled_lambda = CombinedContext.principled_lambda(alpha, N)
    principled_Lambda = principled_lambda / (1 + principled_lambda)
    print(f"  alpha={alpha}, N={N}  ->  lambda={principled_lambda:.6f}  "
          f"->  Lambda={principled_Lambda:.6f}")
    total_err = 0
    for seq in seqs:
        ctx = CombinedContext(ctx_dim=512, input_dim=M, m=22, Lambda=principled_Lambda,
                               alpha=alpha, seed=3)
        mach = SequenceMachine2005(alphabet_size, ctx, input_M=M, input_N=N,
                                    addr_dim=addr_dim, addr_N=N, seed=3)
        total_err += count_errors(mach, seq)
    print(f"  total_errors at principled Lambda={principled_Lambda:.4f}: {total_err}")


if __name__ == "__main__":
    main()
