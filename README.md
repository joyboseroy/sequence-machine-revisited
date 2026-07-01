# sequence-machine-revisited

A faithful reimplementation of the on-line sequence machine from Joy Bose's
PhD thesis (University of Manchester, 2007), benchmarked against modern sequence models twenty years later.

## Source material

- Bose, Furber, Shapiro. "An associative memory for the on-line recognition
  and prediction of temporal sequences." IJCNN 2005.
- Bose, Furber, Shapiro. "A spiking neural sparse distributed memory
  implementation for learning and predicting temporal sequences." ICANN 2005.
- Bose, Furber, Cumpstey. "An asynchronous spiking neural network which can
  learn temporal sequences." UK Async Forum 2005.
- Bose, Furber, Shapiro. "A system for transmitting a coherent burst of
  activity through a network of spiking neurons." WIRN 2005.
- Bose, J. "Engineering a sequence machine through spiking neurons employing
  rank-order codes." PhD thesis, University of Manchester, 2007.

## Repo structure

```
src/
  sequence_machine_2005.py      abstract sequence machine (thesis Ch 4-5)
  wheel_spiking_machine.py      full spiking sequence machine (thesis Ch 7)
  rag_baselines.py              RAG/vector-DB baselines for experiment 04
  sdm_library/                  modular SDM library — all four variants
    __init__.py
    base.py                     shared interface + encoding utilities
    standard_sdm.py             (a) binary N-of-M SDM (thesis Ch 3)
    rankorder_sdm.py            (b) rank-order significance-vector SDM (Ch 4)
    wheel_sdm.py                (c) wheel-model spiking SDM (Ch 6-7)
    rdlif_sdm.py                (d) RDLIF spiking SDM (Ch 6 / 2005 papers)

experiments/
  01_reproduce_original_figure6.py    sanity check vs 2005 paper Fig 6
  02_parameter_sweep.py               Lambda sweep, principled lambda check
  03_modern_baselines.py              LSTM / Transformer under same protocol
  04_associative_recall_benchmark.py  SDM vs RAG/vector-DB under interference
  05_wheel_spiking_machine.py         wheel-model spiking sequence machine
  06_error_characterization.py        error-type breakdown (thesis sec 9.5.2)
  07_sdm_library_comparison.py        all four SDM variants head-to-head
  08_binary_vs_rankorder_sequence.py  Standard vs RankOrder on sequences

related-work/
  ajwani-lalan-sdm-nengo-2021/        Bernstein 2021 reproduction (see below)
```

## Setup

```
pip install torch numpy nengo
python3 experiments/01_reproduce_original_figure6.py
# ... through ...
python3 experiments/08_binary_vs_rankorder_sequence.py
```

---

## SDM Library (`src/sdm_library/`)

A modular, reusable Python library implementing all four SDM variants from
the thesis, sharing a common interface so they are directly swappable.

**Architecture note — four distinct N-of-M parameters (thesis Figure 3.2):**

The thesis defines four *separate* N-of-M codes, not two. Getting this right
was a real correction made during development:

| Parameter | Notation | Example | Meaning |
|---|---|---|---|
| `N_i`-of-`D` | i-of-A | 11-of-256 | Input address code sparsity |
| `N_a`-of-`D` | a-of-A | 20-of-256 | Address decoder weight sparsity — **separate from N_i** |
| `N_w`-of-`W` | w-of-W | 16-of-4096 | Address decoder output sparsity |
| `N_d`-of-`D` | d-of-D | 11-of-256 | Data code sparsity |

`N_a` controls how many of the `D` inputs each of the `W` address decoder
neurons is connected to. It is a separate parameter from `N_i` — the
Ajwani et al. 2021 paper uses `N_a=20` while `N_i=N_d=11`. The earlier
version of this implementation collapsed `N_a` and `N_i` into one value,
which was wrong.

**Usage:**

```python
from sdm_library import StandardSDM, RankOrderSDM, WheelSDM, RDLIFSDM

# all four share the same interface
sdm = RankOrderSDM(D=256, N_i=11, N_a=20, W=4096, N_w=16, N_d=11, alpha=0.99)
sdm.write(address_vec, data_vec)
recalled = sdm.read(address_vec)
result   = sdm.capacity_test(address_codes, data_codes)
```

| Class | Encoding | Learning rule | Neuron model | Thesis ref |
|---|---|---|---|---|
| `StandardSDM` | Binary N-of-M, OR weights | Logical-OR | — (abstract) | Ch 3 / Furber et al. |
| `RankOrderSDM` | Significance vectors, MAX weights | MAX outer product | — (abstract) | Ch 4 / Fig 4.2 |
| `WheelSDM` | Significance vectors | MAX outer product | Wheel/firefly (closed-form) | Ch 6–7 |
| `RDLIFSDM` | Significance vectors | MAX outer product | RDLIF (numerical ODE) | Ch 6 / 2005 papers |

**Key results from experiment 07:**

- `WheelSDM` and `RankOrderSDM` are **exactly equivalent** (similarity 1.0000
  on every pair), confirming the thesis Chapter 7 claim of exact equivalence
  between the wheel-model spiking and abstract versions.
- `RDLIFSDM` is near-equivalent under low load but diverges significantly
  under interference — exactly the limitation the thesis documents on p.125
  as the reason the wheel model was chosen for the full spiking machine.
- `RankOrderSDM` clearly outperforms `StandardSDM` at capacity (154/160 vs
  132/160 at high load with `W=256`), but both are equivalent under normal
  operating conditions with the full thesis-default `W=4096` decoder — the
  capacity advantage only shows when the memory is saturated.

---

## The spiking sequence machine (Wheel model, thesis Chapter 7)

`src/wheel_spiking_machine.py` implements the full *spiking* sequence machine
the thesis builds in Chapter 7 — not the RDLIF model from the conference
papers.

This distinction matters: Chapter 6 investigates RDLIF specifically for the
complete sequence machine and concludes it **cannot** be tuned to reproduce
the abstract model's firing-order behaviour exactly (p.125). The thesis then
introduces the wheel model *specifically* because it can, stating: "we will
use the wheel model to simulate the complete spiking machine in the next
chapter" (p.128). RDLIF is right for the burst-stability work (Ajwani
subfolder), but the wrong choice for the full machine.

**Wheel model** (thesis eq. 6.10–6.13): phase rises at constant slope *m*,
jumps instantaneously by `weight × significance` on each incoming spike,
fires at threshold Θ. Piecewise-linear dynamics give an exact closed-form
time-to-fire — no timestep simulation needed, making the full
encoder → context → address decoder → data store pipeline tractable.

**Results — spiking vs. abstract model on same sequences (thesis sec 7.7):**

| length | spiking errors | abstract errors |
|---|---|---|
| 10–50 | 0 | 0 |
| 75 | 4 | 0 |
| 100 | 9 | 0 |

Near-perfect equivalence to length 50, then a real divergence at 75–100.
Most likely a capacity-budget effect: the spiking model uses a smaller
address decoder (128 slots) and context dimension (96) than the abstract
model (2048/512) to keep computation fast. Scaling up is the natural next
experiment.

**Documented simplification:** the data store write phase uses instantaneous
max-Hebbian (not literal STDP over a continuous write window). Everything
else — encoding, context update, address decode, read/decode — runs through
genuine wheel-model spiking dynamics.

---

## Related work: Ajwani et al. 2021 (`related-work/ajwani-lalan-sdm-nengo-2021/`)

Reproduction of the 2021 Bernstein Conference successor paper by a
colleague's student at BITS Pilani, with Joy Bose as co-author:

> Ajwani, Lalan, Sen Bhattacharya, Bose (2021). "Sparse Distributed Memory
> using Spiking Neural Networks on Nengo."

Both repos reimplement the same root lineage (Furber/Kanerva N-of-M SDM),
applied to different problems: sequence prediction here, raw memory capacity
there. See that folder's own README for full details and results.

---

## Error-type characterization (thesis Section 9.5.2)

`experiments/06_error_characterization.py` implements a future-work item the
thesis names but never carries out: classifying wrong predictions as
near-miss (correct symbol ranked #2/#3), low-activity (no confident guess),
or confident-wrong (high-confidence but wrong).

Run at reduced memory budget (addr_dim=512) to force failures:

| length | top-1 acc | top-3 acc | near_miss | confident_wrong |
|---|---|---|---|---|
| 50–200 | 100% | 100% | 0 | 0 |
| 400 | 96.5% | 100% | 14 | 0 |
| 800 | 77.5% | 95.6% | 145 | 35 |
| 1200 | 54.7% | 85.9% | 374 | 169 |
| 2000 | 28.2% | 61.9% | 675 | 761 |

There is a **graceful-to-catastrophic transition**: at moderate overload,
errors are almost entirely near-misses (top-3 stays at 95%+ while top-1
drops to 77%). Past length 2000, confident-wrong errors *exceed* near-misses
for the first time — the failure mode flips from "fuzzy but reasonable" to
"confidently wrong." `low_activity` stays at zero throughout: this memory
never signals uncertainty, which is a real risk characteristic for any
agent-memory application.

---

## Brainstorming: thesis future work (Sec 9.4/9.5) mapped to 2026

- 9.4.1 (text prompter) → continual next-token learning without fine-tuning;
  experiments 03/04 already probe this angle.
- 9.4.4 (robot gesture copying) → few-shot imitation learning; one-shot
  association from a single demonstration is still an open problem.
- 9.5.4 (STDP instead of max-Hebbian) → local biologically-plausible
  learning rules (forward-forward, predictive coding); the biggest documented
  simplification across this whole repo.
- 9.5.6 (probability-vector significance) → structurally a primitive attention
  mechanism; comparing it directly against softmax attention on identical data
  would be a clean bridge to the "rank-order attention" thread.
- 9.5.7 (dynamic λ) → learned gating; conceptual ancestor of LSTM forget gates
  and input-dependent attention weighting.
- 9.4.6 (Thorpe's SpikeNET) → modern descendant is spiking transformers and
  neuromorphic vision models (Loihi, SpikingJelly).

---

## Findings

**1. Thesis-faithful reimplementation.**
The combined-model context formula (thesis eq. 5.4/5.5) is one architecture
parameterized by Λ — not three separate models as the conference papers imply.
The corrected implementation gets near-perfect recall across Λ=0.3–0.9 with
zero tuning, and the thesis's own principled λ=α^(N+1) choice gives 2 errors
out of ~180 predictions without any sweep.

**2. One-shot learning beats gradient-based models under the same protocol.**

| seq length | SDM (combined, Λ=0.5) | LSTM | Transformer |
|---|---|---|---|
| 30 | 0 | 25 | 3 |
| 60 | 0 | 41 | 23 |
| 90 | 0–1 | 69 | 34 |

Hebbian one-shot writes vs. single SGD steps — not surprising, but a clean,
concrete demonstration of why one-shot associative memory is a *different*
tool from gradient-trained sequence models, not a worse one.

**3. SDM degrades gracefully; a budget-capped vector store fails catastrophically.**

| distractors | SDM (2048-slot) | UnboundedVectorDB | CappedVectorDB (2048-slot, FIFO) |
|---|---|---|---|
| 0–1000 | ✓ | ✓ | ✓ |
| 5000 | ✓ | ✓ | ✗ |
| 20000 | ✓ | ✓ | ✗ |

The real finding: once a memory budget is enforced, the SDM's max-Hebbian
write degrades gracefully under address collisions, while a FIFO-capped
vector store fails permanently the moment the cue is evicted — a hard cliff
vs. graceful degradation. `UnboundedVectorDB` never fails by construction,
which is itself a caveat about what "RAG memory" usually means in practice.

**4. All four SDM variants confirmed correct.**
`StandardSDM` and `RankOrderSDM` are functionally identical under normal
conditions (both 100% at W=4096); the capacity difference only shows when
saturated. `WheelSDM` exactly matches `RankOrderSDM` (sim=1.0000 on every
pair). `RDLIFSDM` diverges under interference, as the thesis documents.

---

## Honest limitations

- All results are single runs, not averaged over multiple seeds with error
  bars. The thesis averages over 5–30 runs; paper-grade numbers need this.
- LSTM/Transformer hyperparameters in experiment 03 are not tuned — a
  reviewer could argue the baselines were not given a fair chance.
- FIFO eviction in `CappedVectorDB` is the simplest possible policy. The
  graceful-vs-catastrophic finding should be tested against LRU eviction
  before claiming it generalises to vector stores as a category.
- RAG baselines use random projections as embeddings, not real learned
  embeddings (e.g. sentence-transformers). Appropriate for isolating the
  retrieval mechanism, but not comparable to literature RAG numbers.
- No real-world dataset yet — everything is synthetic random sequences.
- STDP learning rule not implemented; max-Hebbian is used throughout as
  a documented simplification.

## Suggested next steps

1. Average experiments 01, 03, 04 over multiple seeds with error bars.
2. Add Mamba/RWKV as baselines in experiment 03.
3. Add LRU-eviction `CappedVectorDB` variant to experiment 04.
4. Scale the wheel-model spiking machine (experiment 05) to match the
   abstract model's address decoder size and re-test the equivalence claim.
5. Write up findings 2 and 3 above as the empirical core of a short arXiv
   paper — thesis-faithful reimplementation, real RAG baseline, honest limits.
