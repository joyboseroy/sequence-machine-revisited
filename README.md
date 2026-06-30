# sequence-machine-revisited

A faithful PyTorch reimplementation of the on-line sequence machine from
my PhD work (University of Manchester, 2005),
benchmarked against modern sequence models (LSTM, Transformer) and a
dense-memory baseline, twenty years later.

## Source material

The architecture here is reimplemented directly from:

- Bose, Furber, Shapiro. "An associative memory for the on-line recognition
  and prediction of temporal sequences." IJCNN 2005.
- Bose, Furber, Shapiro. "A spiking neural sparse distributed memory
  implementation for learning and predicting temporal sequences." ICANN 2005.
- Bose, Furber, Cumpstey. "An asynchronous spiking neural network which can
  learn temporal sequences." UK Async Forum 2005.
- Bose, Furber, Shapiro. "A system for transmitting a coherent burst of
  activity through a network of spiking neurons." WIRN 2005.

This repo reimplements the abstract (non-spiking) version of the
architecture: rank-ordered N-of-M codes, a modified Kanerva Sparse
Distributed Memory (SDM) with max-based one-shot Hebbian learning, and the
three context-encoding schemes the original papers compared (shift
register, context neural layer, and the "combined" model that won). The
spiking/asynchronous-timing layer described in the WIRN/Async Forum papers
is a separate, lower-level engineering concern not required to reproduce
the *functional* behaviour of the machine, and is not reimplemented here.

## What's in this repo

```
src/sequence_machine_2005.py        core reimplementation
experiments/01_reproduce_original_figure6.py   sanity check vs. the 2005 paper
experiments/02_parameter_sweep.py              small grid search over lambda, x, K
experiments/03_modern_baselines.py             LSTM / Transformer on the SAME protocol
experiments/04_associative_recall_benchmark.py long-distractor recall benchmark
```

## Setup

```
pip install torch
python3 experiments/01_reproduce_original_figure6.py
python3 experiments/02_parameter_sweep.py
python3 experiments/03_modern_baselines.py
python3 experiments/04_associative_recall_benchmark.py
```

## Findings so far

**1. The reimplementation behaves like the original.** Errors stay near
zero while the sequence length is shorter than the alphabet (15 symbols),
then climb once repeats are forced — exactly the qualitative pattern
described in the IJCNN paper. The "combined" context model (shift register
+ context layer) wins after a small parameter sweep, matching the original
paper's conclusion, though I haven't yet matched their exact recall numbers
— my "context neural layer" model in particular is a weaker approximation
of theirs and underperforms across all λ; this is a known gap, not a
hidden result.

**2. Under one-shot online learning, the 2005 machine beats modern
gradient-based models head-to-head.** Forcing an LSTM and a small causal
Transformer into the *same* protocol the SDM was built for — single online
pass, one gradient step per symbol, no epochs, no pretraining — they get
meaningfully more next-symbol prediction errors than the SDM at every
sequence length tested:

| seq length | SDM (combined, tuned) | LSTM | Transformer |
|---|---|---|---|
| 30 | 0 | 25 | 3 |
| 60 | 0 | 41 | 23 |
| 90 | 26 | 69 | 34 |

This isn't surprising once you think about it (Hebbian one-shot writes vs.
single SGD steps), but it's a clean, concrete illustration of *why*
one-shot associative memory is a different tool than gradient-trained
sequence models, not a worse one — which is the actual thesis worth writing
up.

**3. Long-distractor associative recall: SDM holds, dense MLP collapses.**
Store one cue→target association, then write D unrelated distractor
associations, then probe the original cue:

| distractors | SDM | Dense MLP (retrained, 50 epochs) |
|---|---|---|
| 0 | ✓ | ✓ |
| 10 | ✓ | ✓ |
| 50 | ✓ | ✓ |
| 200 | ✓ | ✗ |
| 1000 | ✓ | ✗ |
| 5000 | ✓ | ✗ |

This is the strongest and most "publishable" result of the four
experiments — it's the concrete version of the "ABCDEF → XYZ after 10,000
distractions" benchmark idea, and the gap is large and clean.

## Honest limitations / what's not done yet

- The context-layer model's exact match to the paper's λ-tuned numbers is
  unresolved — my "neural layer" scramble matrix is a rough approximation,
  not derived from the thesis's actual context-layer math. Worth revisiting
  once the full thesis PDF's relevant chapter is read closely.
- No Mamba/RWKV/MemGPT/RAG baselines yet (Theme 1/2 from the brainstorm) —
  the LSTM/Transformer comparison here is meant to establish the protocol
  is sound before adding more baselines.
- The dense MLP baseline in experiment 4 is a deliberately simple "dense
  memory" stand-in, not MemGPT or a real vector-DB RAG pipeline; swapping
  in a proper RAG baseline would make the associative-recall result more
  defensible for a paper submission.
- No real-world dataset yet (music/EEG/robot trajectories/stock prices, per
  Theme 5) — everything here is synthetic random sequences over a small
  alphabet, matching the original paper's own test protocol.

## Suggested next steps

1. Tighten the context-layer model against the actual thesis math (not just
   the conference-paper summary) so all three context models are
   apples-to-apples.
2. Add Mamba/RWKV as baselines in experiment 3 — they're the most natural
   "linear-recurrent-state" comparison to the SDM's context mechanism.
3. Swap the dense-MLP baseline in experiment 4 for a real RAG/vector-DB
   pipeline, which is the actual claim worth making for arXiv: does SDM
   outperform vector-database retrieval for long-term agent memory under
   interference?
4. Write up findings 2 and 3 above as the empirical core of a short paper —
   they're the parts of the original brainstorm (Themes 9 and 14) that are
   now backed by actual numbers rather than hypothesis.
