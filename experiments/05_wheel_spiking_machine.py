"""
Test the wheel-model spiking sequence machine on the thesis's own
illustrative example: the repeated sequence 7,1,5,1,7,1,5,1,... (used in
Fig 7.7 / 8.3 of the thesis and the WIRN paper's Fig 3), where "1" needs
context (preceded by either 7 or 5) to predict its successor correctly.
Also runs a random-sequence test matching experiment 01's protocol, for a
side-by-side comparison against the abstract (non-spiking) model.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
from wheel_spiking_machine import WheelSpikingSequenceMachine


def thesis_example():
    print("=== Thesis Fig 7.7 example: sequence 7,1,5,1,7,1,5,1,7,1,5,1 ===")
    seq = [7, 1, 5, 1] * 3
    machine = WheelSpikingSequenceMachine(num_symbols=10, seed=42)
    preds = machine.run_sequence(seq)
    print("input:      ", seq)
    print("predictions:", preds, " (prediction[i] is the model's guess for symbol i+1)")
    correct_after_first_pass = sum(
        1 for i in range(4, len(seq) - 1) if preds[i] == seq[i + 1]
    )
    total_after_first_pass = len(seq) - 1 - 4
    print(f"correct predictions after the first 7,1,5,1 pass: "
          f"{correct_after_first_pass}/{total_after_first_pass}")


def random_sequence_test():
    print("\n=== Random sequence test (alphabet=15, matching experiment 01 protocol) ===")
    alphabet_size = 15
    for length in [10, 20, 30, 50]:
        rng = random.Random(length)
        seq = [rng.randrange(alphabet_size) for _ in range(length)]
        machine = WheelSpikingSequenceMachine(num_symbols=alphabet_size, seed=7)
        # single online pass: learn, then re-run from scratch's context
        # but SAME learned weights, to test recall (matching abstract model's protocol)
        machine.run_sequence(seq)  # first pass: learns
        machine.context_events = []  # reset context only, KEEP learned W_data
        errors = 0
        for i, sym in enumerate(seq):
            pred = machine.step(sym)
            if i + 1 < len(seq) and pred != seq[i + 1]:
                errors += 1
        print(f"  len={length:3d}  errors={errors:3d} / {len(seq)-1}")


def equivalence_check():
    """Thesis Section 7.7: 'Verification of equivalence with the time
    abstracted model.' Run the SAME random sequences through both the
    wheel-model spiking machine and the abstract (non-spiking) combined
    model from experiment 01, and compare error counts directly."""
    print("\n=== Spiking (wheel model) vs abstract model, same sequences ===")
    from sequence_machine_2005 import SequenceMachine2005, CombinedContext

    alphabet_size = 15
    M, N = 256, 11
    for length in [10, 20, 30, 50, 75, 100]:
        rng = random.Random(length)
        seq = [rng.randrange(alphabet_size) for _ in range(length)]

        spiking = WheelSpikingSequenceMachine(num_symbols=alphabet_size, seed=7)
        spiking.run_sequence(seq)
        spiking.context_events = []
        sp_errors = 0
        for i, sym in enumerate(seq):
            pred = spiking.step(sym)
            if i + 1 < len(seq) and pred != seq[i + 1]:
                sp_errors += 1

        ctx = CombinedContext(ctx_dim=512, input_dim=M, m=22, Lambda=0.5, seed=3)
        abstract = SequenceMachine2005(alphabet_size, ctx, input_M=M, input_N=N,
                                        addr_dim=2048, addr_N=N, seed=3)
        abstract.run_sequence(seq)
        abstract.reset()
        ab_errors = 0
        for i, sym in enumerate(seq):
            pred = abstract.step(sym)
            if i + 1 < len(seq) and pred != seq[i + 1]:
                ab_errors += 1

        print(f"  len={length:3d}  spiking_errors={sp_errors:3d}  abstract_errors={ab_errors:3d}")


if __name__ == "__main__":
    thesis_example()
    random_sequence_test()
    equivalence_check()
