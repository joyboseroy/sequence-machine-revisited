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

## The actual spiking implementation (Wheel model, thesis Chapter 7)

`src/wheel_spiking_machine.py` and `experiments/05_wheel_spiking_machine.py` implement the full *spiking* sequence machine the thesis actually builds in Chapter 7 — not the RDLIF model from the conference papers and from `related-work/ajwani-lalan-sdm-nengo-2021/`.

This distinction matters and is easy to miss: Chapter 6 of the thesis investigates RDLIF (the model in the IJCNN/ICANN/WIRN papers) specifically for the complete sequence machine, and concludes it **cannot** be tuned to reproduce the abstract model's rank-order/significance-vector behaviour exactly — "implementing the temporal abstraction by the RDLIF model... is not feasible" (p.125). The thesis then introduces the **wheel (firefly/spin) model** specifically because it *can* be tuned to match the abstract model exactly, and states plainly: "we will use the wheel model to simulate the complete spiking machine in the next chapter" (p.128). So a faithful "spiking version of the sequence machine" means the wheel model, not RDLIF — RDLIF is the right choice for the separate burst-stability questions in the Ajwani-reproduction subfolder, but the wrong choice here.

**Wheel model mechanics** (thesis eq 6.10–6.13): each neuron has a phase that rises at constant slope *m* once active, and jumps instantaneously by `connection_weight × input_significance` on each incoming spike; it fires when its phase crosses threshold Θ. Because this is piecewise-linear, the time-to-fire has an exact closed-form solution given the sequence of incoming spike events — this implementation uses that closed form directly rather than timestep simulation, which is what makes the full architecture (encoder → context layer → address decoder → data store → decode) tractable to run faithfully in this session.

One real, documented bug fix along the way: the first version used `threshold=1.0` with weights in [0,1], so a single incoming spike could already cross threshold — firing order became essentially noise rather than meaningfully reflecting the weighted, rank-ordered input. Setting the threshold high enough that no neuron crosses *mid-burst* fixed this: firing order is then determined entirely by the post-burst ramp phase, which makes it exactly equal to "rank by total weighted significance" — precisely the equivalence to the abstract model's top-k decode that the thesis is going for. Random-sequence recall went from near-chance to perfect after this one change.

**Results: spiking vs. abstract model, same sequences (thesis Section 7.7's "verification of equivalence")**

| sequence length | spiking (wheel model) errors | abstract model errors |
|---|---|---|
| 10 | 0 | 0 |
| 20 | 0 | 0 |
| 30 | 0 | 0 |
| 50 | 0 | 0 |
| 75 | 4 | 0 |
| 100 | 9 | 0 |

Near-perfect equivalence through length 50, then a real, honest divergence: the spiking version starts accumulating errors at 75–100 while the abstract model (with its much larger 2048-slot address decoder) stays perfect. This is very likely a capacity-budget effect, not a fundamental inequivalence — the wheel-model implementation here uses a deliberately small address decoder (128 slots) and context dimension (96) to keep the closed-form computation fast, against the abstract model's 2048/512. That's the natural next experiment: scale the spiking model's layer sizes up to match and see whether the divergence closes, which would support the thesis's claim of exact equivalence; if it doesn't fully close, that's an interesting finding in its own right.

On the thesis's own illustrative example sequence (`7,1,5,1` repeated three times, used in Fig 7.7/8.3), the spiking model gets 5 of 7 post-learning predictions correct — not perfect, plausibly due to the `decode_to_symbol` function's overlap-based nearest-match having ties on such a short, repetitive sequence; worth a closer look if this becomes the basis for a paper claim rather than a demonstration.

**Documented simplification**: the data store's write phase uses the same instantaneous max-Hebbian rule as the abstract model, not literal spike-timing-dependent plasticity over a continuous write window (thesis Section 7.4.3's timing constraints in more depth than was practical here). Everything else — encoding, the combined-model context update, address decoding, and the read/decode operation — runs through genuine wheel-model spiking dynamics with real closed-form fire-time computation and significance-vector propagation between layers.



`related-work/ajwani-lalan-sdm-nengo-2021/` contains a reproduction of the
2021 Bernstein Conference successor paper from the same lineage of work:

> Ajwani, Lalan, Sen Bhattacharya, Bose (2021). "Sparse Distributed Memory
> using Spiking Neural Networks on Nengo."

This is the natural follow-on study by a colleague's student (BITS Pilani),
with Joy Bose as co-author, building on the same N-of-M / Sparse
Distributed Memory architecture this repo reimplements for the sequence
machine. It belongs alongside the main sequence-machine work rather than
as a separate repo, since both are reproductions of the same root
intellectual lineage (Furber/Kanerva-style SDM with N-of-M codes), just
applied to different problems (sequence prediction here vs. raw memory
capacity there). See that folder's own README for details, results, and
its own honest-limitations section.

## Setup

```
pip install torch
python3 experiments/01_reproduce_original_figure6.py
python3 experiments/02_parameter_sweep.py
python3 experiments/03_modern_baselines.py
python3 experiments/04_associative_recall_benchmark.py
```

## Findings so far

**1. The reimplementation now matches the original thesis precisely, not just the conference papers.** The conference papers (IJCNN/ICANN/WIRN 2005) describe the "shift register," "context neural layer," and "combined" context models as if they were three separate architectures. Reading PhD thesis Chapter 5 ("Designing a sequence machine," Bose 2007) directly shows this isn't accurate: there is **one** combined-model formula (eq. 5.4/5.5),

```
C_n = nof_m( Lambda * scale(P1 @ C_{n-1}) + (1-Lambda) * scale(P2 @ I_n), m, M, alpha )
```

parameterized by a single modulation factor (λ, or its bounded form Λ = λ/(1+λ)), which smoothly interpolates between input-dominated ("shift-register-like") and context-dominated ("neural-layer-like") behaviour. My first-pass implementation invented a separate, ad-hoc "context neural layer" class that doesn't correspond to anything in the actual thesis — it has been removed. The corrected, thesis-faithful `CombinedContext` class is now the only context model alongside the literal explicit-window `ShiftRegisterContext` from Chapter 2, which the thesis treats as a simpler baseline for comparison, not a competing version of the combined model.

Two other corrections from the thesis text, not in the conference papers:
- `scale()` is L1 normalisation (components sum to 1) applied to both the projected old context and the projected input *before* combining them — my first pass combined raw, unnormalised values.
- The top-K selection step (`nof_m`) doesn't just mask non-selected components to zero and keep raw magnitudes — it **re-encodes the selected components as a proper rank-order code** (weights 1, α, α², ... by relative rank), exactly like the input encoder. My first pass kept raw values, which is a different (and less principled) operation.
- The thesis also gives a non-arbitrary, principled choice for the modulation factor: λ = α^(N+1), tying it directly to the significance ratio α and the number of active input components N, rather than treating it as a free hyperparameter to sweep blindly.

**2. With the corrected formula, the sequence machine gets near-perfect recall across nearly the entire useful Λ range (0.3–0.9), and the thesis's own "principled" λ=α^(N+1) choice lands close to optimal (2 errors out of ~180 test predictions) without any tuning at all:**

```
Lambda=0.0  total_errors=127   (degenerate: no memory of past at all)
Lambda=0.1  total_errors=20
Lambda=0.2  total_errors=2
Lambda=0.3–0.9  total_errors=0
principled Lambda=0.2202 (from alpha=0.9, N=11): total_errors=2
```

This is a much cleaner and more convincing match to the original paper's claimed near-perfect recall than my first pass achieved, and it required no hand-tuning — just implementing the actual formula correctly.

**3. The literal shift register (Chapter 2's explicit fixed-window baseline) clearly underperforms the combined model**, exactly as the thesis claims it should, with errors growing from single digits to ~16 as sequence length increases and forces more repeated symbols past the lookback window.

**4. Under one-shot online learning, the 2005 machine beats modern gradient-based models head-to-head.** Forcing an LSTM and a small causal Transformer into the *same* protocol the SDM was built for — single online pass, one gradient step per symbol, no epochs, no pretraining — they get meaningfully more next-symbol prediction errors than the SDM at every sequence length tested:

| seq length | SDM (combined, Λ=0.5) | LSTM | Transformer |
|---|---|---|---|
| 30 | 0 | 25 | 3 |
| 60 | 0 | 41 | 23 |
| 90 | 0–1 | 69 | 34 |

This isn't surprising once you think about it (Hebbian one-shot writes vs. single SGD steps), but it's a clean, concrete illustration of *why* one-shot associative memory is a different tool than gradient-trained sequence models, not a worse one — which is the actual thesis worth writing up.

**5. Long-distractor associative recall: SDM degrades gracefully, a budget-capped vector store fails catastrophically — and the comparison is now against a real RAG baseline, not a strawman.** The first version of this experiment used a dense MLP retrained from scratch as the "non-SDM" comparison, which conflated gradient-training capacity with retrieval mechanism and wasn't representative of how RAG/vector-DB agent memory actually works in practice. It's been replaced with two honest RAG baselines in `src/rag_baselines.py`: an `UnboundedVectorDB` (the standard naive setup — store everything, retrieve by nearest-neighbour cosine similarity, no capacity limit) and a `CappedVectorDB` with the SAME memory budget as the SDM (2048 slots, matching the SDM's address-decoder dimensionality) using FIFO eviction once full — the fair, apples-to-apples comparison.

Store one cue→target association, then write D unrelated distractor associations, then probe the original cue:

| distractors | SDM (2048-slot budget) | UnboundedVectorDB (no budget) | CappedVectorDB (2048-slot budget, FIFO) |
|---|---|---|---|
| 0 | ✓ | ✓ | ✓ |
| 10 | ✓ | ✓ | ✓ |
| 50 | ✓ | ✓ | ✓ |
| 200 | ✓ | ✓ | ✓ |
| 1000 | ✓ | ✓ | ✓ |
| 5000 | ✓ | ✓ | ✗ |
| 20000 | ✓ | ✓ | ✗ |

The actual finding here is not "SDM beats vector DBs" — `UnboundedVectorDB` never fails, by construction, since unbounded exact storage with exact retrieval is essentially unbeatable, which is itself an important caveat about what "RAG memory" usually means when people don't think about its eventual storage/context budget. The real finding is in the **shape of failure** once a memory budget is enforced: the SDM, given the *same* 2048-slot budget as `CappedVectorDB`, never fails across any distractor count tested (its max-based Hebbian write degrades gracefully under address collisions rather than destroying old associations outright), while the FIFO-capped vector store fails completely and permanently the moment the cue gets evicted past the budget (a hard cliff, not graceful degradation). This is the actual, defensible claim for a paper: fixed-capacity content-addressable memory (SDM) degrades gracefully under interference, while fixed-capacity vector-store memory with a naive eviction policy degrades catastrophically once its budget is exceeded.

## Honest limitations / what's not done yet

- Results above are single runs per length, not averaged over multiple seeds with error bars (the thesis itself averages over 5–30 runs, e.g. Fig 8.4/8.6). For paper-grade numbers this needs to be repeated across seeds.
- The LSTM/Transformer hyperparameters in experiment 03 (hidden=64, lr=0.05, one SGD step) are reasonable defaults, not tuned — a reviewer could fairly ask whether the baselines were given a real chance under this protocol.
- No Mamba/RWKV/MemGPT baselines yet (Theme 1/2 from the original brainstorm) — the RAG/vector-DB comparison (item 5 above) is done; a true MemGPT-style hierarchical/summarising memory would be a further, harder comparison.
- `CappedVectorDB`'s eviction policy is plain FIFO, the simplest possible policy and arguably a strawman in its own right (a real system might use LRU, importance-weighted eviction, or periodic summarisation). The graceful-vs-catastrophic finding (item 5) should be re-tested against at least LRU eviction before claiming it generalises to "vector stores" as a category rather than "naive FIFO-capped vector stores" specifically.
- The embeddings in `rag_baselines.py` are random projections of the same N-of-M codes used elsewhere, not a real learned embedding model (e.g. sentence-transformers) — appropriate for isolating the retrieval/storage *mechanism* from embedding quality, but worth flagging if this gets compared against literature numbers that use real embeddings.
- No real-world dataset yet (music/EEG/robot trajectories/stock prices, per Theme 5) — everything here is synthetic random sequences over a small alphabet, matching the original paper's own test protocol.
- Chapters 6–8 of the thesis (the spiking-neuron implementation and its specific timing/noise tests) haven't been mined yet — they're a separate, lower-level concern from the functional architecture reimplemented here, but could matter for a "spiking transformer" follow-on (Theme 4/10 from the brainstorm).

## Suggested next steps

1. Average results 1, 2, 4 above over multiple seeds with error bars (matching the thesis's own 5–30 run averaging), now that the underlying model is correct.
2. Add Mamba/RWKV as baselines in experiment 3 — they're the most natural "linear-recurrent-state" comparison to the SDM's context mechanism.
3. Add an LRU-eviction `CappedVectorDB` variant to test whether the graceful-vs-catastrophic finding in item 5 holds against a less naive eviction policy than FIFO, before claiming the result generalises.
4. Write up findings 4 and 5 above as the empirical core of a short paper — they're the parts of the original brainstorm (Themes 9 and 14) that are now backed by actual numbers, on a thesis-faithful reimplementation, against a real RAG baseline rather than a strawman.
