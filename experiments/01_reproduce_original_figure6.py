"""
Sanity check: reproduce (in spirit) Figure 6 of the IJCNN 2005 paper —
comparing shift register / context layer / combined model on prediction
error as a function of sequence length, with an alphabet of 15 symbols.

This validates that the reimplementation behaves the way the 2005 paper
describes (combined model should make fewer errors than the other two,
especially as sequences get longer and force more repeated symbols).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import random
from sequence_machine_2005 import (
    SequenceMachine2005, ShiftRegisterContext, CombinedContext
)


def random_sequence(alphabet_size, length, seed):
    rng = random.Random(seed)
    return [rng.randrange(alphabet_size) for _ in range(length)]


def count_errors(machine, seq):
    """On-line single pass: present sequence once (machine learns as it
    goes), then present it again and count how many of the *next-symbol*
    predictions are wrong."""
    machine.run_sequence(seq)  # first pass: learns
    machine.reset()
    errors = 0
    for i, sym in enumerate(seq):
        pred = machine.step(sym)
        if i + 1 < len(seq):
            target = seq[i + 1]
            if pred != target:
                errors += 1
    return errors


def main():
    alphabet_size = 15
    lengths = list(range(10, 101, 10))
    M, N = 256, 11
    addr_dim = 2048

    results = {"shift_register": [], "combined_input_dominated": [], "combined_balanced": []}

    for length in lengths:
        seq = random_sequence(alphabet_size, length, seed=length)

        sr_ctx = ShiftRegisterContext(M=M, lookback=2, decay=0.5)
        machine_sr = SequenceMachine2005(alphabet_size, sr_ctx, input_M=M,
                                          input_N=N, addr_dim=addr_dim, addr_N=N, seed=1)
        results["shift_register"].append(count_errors(machine_sr, seq))

        # Lambda=0.5 == lambda=1.0 in thesis eq 5.4: input and past context
        # equally important. Thesis calls this the "neural layer" case.
        cl_ctx = CombinedContext(ctx_dim=512, input_dim=M, m=22, Lambda=0.5, seed=2)
        machine_cl = SequenceMachine2005(alphabet_size, cl_ctx, input_M=M,
                                          input_N=N, addr_dim=addr_dim, addr_N=N, seed=2)
        results["combined_balanced"].append(count_errors(machine_cl, seq))

        # Lower Lambda: input-dominated, gradual forgetting (shift-register-like)
        cb_ctx = CombinedContext(ctx_dim=512, input_dim=M, m=22, Lambda=0.2, seed=3)
        machine_cb = SequenceMachine2005(alphabet_size, cb_ctx, input_M=M,
                                          input_N=N, addr_dim=addr_dim, addr_N=N, seed=3)
        results["combined_input_dominated"].append(count_errors(machine_cb, seq))

        print(f"len={length:3d}  shift_register={results['shift_register'][-1]:3d}  "
              f"combined(Lambda=0.5)={results['combined_balanced'][-1]:3d}  "
              f"combined(Lambda=0.2)={results['combined_input_dominated'][-1]:3d}")

    return lengths, results


if __name__ == "__main__":
    main()
