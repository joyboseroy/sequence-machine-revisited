"""
Error-type characterisation and top-k accuracy, implementing thesis
section 9.5.2's suggested future test: "the chosen symbol may be incorrect
but the output symbol whose activity was second highest or third highest
may be the correct symbol... we can list the various types of errors and
measure what is the most common type of errors."

This experiment classifies every wrong prediction into one of three types:

  NEAR_MISS    -- the correct symbol was ranked #2 or #3 by similarity
                  (the memory "almost" got it right; context discrimination
                  was imprecise, not absent)
  LOW_ACTIVITY -- the top similarity score is below a threshold, meaning
                  the memory had no confident guess at all (thesis: "errors
                  caused due to low activity, with the system unable to
                  make an informed guess")
  CONFIDENT_WRONG -- a high-confidence prediction that is simply incorrect
                  (the worst kind: the memory was sure, and wrong)

This is a genuinely new measurement -- none of experiments 01-05 look past
exact top-1 match, so a sequence with 5 "errors" could mean very different
things (5 near-misses vs 5 confident-wrong predictions), and this is the
first experiment able to tell them apart.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
from sequence_machine_2005 import SequenceMachine2005, CombinedContext


def random_sequence(alphabet_size, length, seed):
    rng = random.Random(seed)
    return [rng.randrange(alphabet_size) for _ in range(length)]


def classify_errors(machine, seq, confidence_threshold=0.3, k=3):
    """Run the sequence once (learning), then again (testing), and for
    every wrong top-1 prediction, classify the error type using the
    top-k ranked candidates."""
    machine.run_sequence(seq)  # learning pass
    machine.reset()

    counts = {"correct": 0, "near_miss": 0, "low_activity": 0, "confident_wrong": 0}
    top1_correct, top3_correct = 0, 0
    total = 0

    for i, sym in enumerate(seq):
        ranked = machine.step_topk(sym, k=k)
        if i + 1 >= len(seq):
            continue
        total += 1
        target = seq[i + 1]
        predicted_symbols = [s for s, _ in ranked]
        top_score = ranked[0][1]

        if predicted_symbols[0] == target:
            counts["correct"] += 1
            top1_correct += 1
            top3_correct += 1
        elif target in predicted_symbols[1:]:
            counts["near_miss"] += 1
            top3_correct += 1
        elif top_score < confidence_threshold:
            counts["low_activity"] += 1
        else:
            counts["confident_wrong"] += 1

    return counts, total, top1_correct, top3_correct


def main():
    alphabet_size = 15
    M, N = 256, 11
    addr_dim = 512  # smaller budget than experiments 01-05 (2048) to actually
                     # force errors in the tested length range, rather than
                     # trivially perfect recall throughout

    print(f"{'len':>5} {'top1_acc':>9} {'top3_acc':>9} {'correct':>8} "
          f"{'near_miss':>10} {'low_activ':>10} {'conf_wrong':>11}")

    for length in [50, 100, 200, 400, 800, 1200, 2000]:
        seq = random_sequence(alphabet_size, length, seed=length)
        ctx = CombinedContext(ctx_dim=512, input_dim=M, m=22, Lambda=0.5, seed=3)
        machine = SequenceMachine2005(alphabet_size, ctx, input_M=M, input_N=N,
                                       addr_dim=addr_dim, addr_N=N, seed=3)
        counts, total, top1, top3 = classify_errors(machine, seq)
        top1_acc = top1 / total
        top3_acc = top3 / total
        print(f"{length:>5} {top1_acc:>9.2%} {top3_acc:>9.2%} "
              f"{counts['correct']:>8} {counts['near_miss']:>10} "
              f"{counts['low_activity']:>10} {counts['confident_wrong']:>11}")

    print("\nInterpretation: a high near_miss count relative to "
          "confident_wrong means the memory IS discriminating context "
          "correctly most of the time and just narrowly mis-ranking close "
          "candidates -- a different (and less concerning) failure mode "
          "than confidently predicting the wrong symbol outright.")


if __name__ == "__main__":
    main()
