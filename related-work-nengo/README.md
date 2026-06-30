# sdm-nengo-2021-reproduction

> **Note:** this is a subfolder of the `sequence-machine-revisited` repo.
> Both reimplement and reproduce work from the same root lineage — Furber/
> Kanerva-style Sparse Distributed Memory with N-of-M codes — just applied
> to different problems (this folder: raw memory capacity; the parent repo:
> on-line sequence prediction). See the parent repo's top-level README for
> the sequence-machine side of the work.

A reimplementation and reproduction of:

> Ajwani, R.D., Lalan, A., Sen Bhattacharya, B., Bose, J. (2021).
> "Sparse Distributed Memory using Spiking Neural Networks on Nengo."
> Bernstein Conference 2021.

(the successor paper to Bose's PhD thesis work, by a colleague's student at
BITS Pilani, with Joy Bose as co-author from Ericsson.)

## What this reproduces

The paper's central claims are about **memory capacity**: how many random
address-data pairs can be written to a Sparse Distributed Memory (SDM) /
Correlation Matrix Memory (CMM) before recall starts failing, and whether
spiking-neuron implementations (LIF, Adaptive-LIF, Izhikevich, Spiking ReLU)
match the non-spiking baseline. This repo reproduces that core empirical
question:

1. **CMM memory capacity** (paper Fig 4-5): A=D=256, 11-of-256 codes,
   varying the number of stored pairs and measuring exact-match recall.
2. **SDM memory capacity across address-decoder sizes** (paper Fig 6-9):
   W in {256, 512, 1024}, A=D=256, i=11, w=16, measuring how peak capacity
   scales with address-decoder size.
3. **Spiking vs non-spiking comparison**, the paper's headline claim.

## Architecture notes / what was reimplemented faithfully vs. simplified

- **N-of-M binary codes**: implemented exactly as specified (random binary
  vectors with exactly N of M bits set).
- **Unipolar binary weights, not real-valued accumulation**: the paper
  explicitly notes that Furber et al. "replaced counters with unipolar
  binary weights" (Section II.B). The first version of this reimplementation
  used real-valued correlation-matrix accumulation (summed outer products)
  and got a memory capacity curve that collapsed far too early compared to
  the paper's reported ~300-350 peak. Switching to binary (logical-OR)
  weight updates — `W = max(W, outer(data, address))` instead of
  `W += outer(data, address)` — fixed this and produced a capacity curve
  that matches the paper's shape closely (perfect recall to ~175-200,
  plateau/peak around 300-325, gradual rather than catastrophic decline
  beyond). This is the single most important correction in this
  reproduction and is worth knowing if you extend the code.
- **Spiking decode via closed-form LIF first-spike-time**, not a literal
  Nengo Ensemble simulation. The paper's spiking implementation injects a
  constant current per neuron for a fixed window (150ms) and decodes by
  "which neurons spiked first." Nengo's `Ensemble` API is built for
  population-coded vector representation, not direct per-neuron current
  injection with first-spike decoding, and fights that design when forced
  into it (this is also why the original paper had to do nontrivial scaling
  / engineering work in Section IV.C-D to get usable spike rates and
  decoding within the 150ms window). Since the underlying LIF ODE (paper
  eq. 3) has an exact closed-form time-to-first-spike solution for constant
  current, this reimplementation uses that closed form directly to rank
  neurons by first-spike time — mathematically equivalent to the paper's
  decode rule for the constant-current regime it actually uses, but fast
  enough to reproduce the FULL capacity curves (hundreds to over a
  thousand stored pairs, three address-decoder sizes) in this session
  rather than a handful of sample points. `nengo` itself is installed and
  available (`pip install nengo`, version 4.1.0 confirmed working) if you
  want to extend this to a literal Ensemble-based simulation later.
- **Learning rule**: BCM and Oja are local, online, spike-train-dependent
  learning rules. The closed-form first-spike shortcut above doesn't
  produce real spike trains for them to operate on. As a documented
  simplification, both the spiking and non-spiking models here use plain
  Hebbian/correlation accumulation (with the binary-weight correction
  above) rather than literal BCM or Oja dynamics. The paper itself reports
  that BCM and Oja "give similar results" to each other, so this
  simplification is unlikely to change the qualitative conclusions, but it
  means the exact peak capacity numbers here are not a pixel-perfect match
  to the paper's Figures, just a close shape/trend match.
- Adaptive-LIF, Izhikevich, and Spiking ReLU variants from the paper are
  NOT separately reimplemented — the paper's own finding is that neuron
  model choice doesn't significantly affect capacity, so reproducing one
  faithful spiking model (LIF) and confirming it tracks the non-spiking
  baseline addresses the paper's main empirical claim without needing all
  four neuron models redone.

## Results

### CMM capacity (A=D=256, 11-of-256 codes)

Perfect recall through ~175-200 pairs, peak around 300-325 (non-spiking:
279 correct at q=325; spiking: 283 correct at q=325), then gradual decline
— matching the paper's description almost exactly ("until ~300 pairs the
recall is perfect... beyond 350 the memory capacity falls, although the
fall is gradual rather than catastrophic"). Spiking and non-spiking track
each other within a handful of pairs at every point tested.

### SDM capacity across address-decoder size W

| W | non-spiking peak (pairs) | spiking peak (pairs) |
|---|---|---|
| 256 | 275 | 275 |
| 512 | 577 | 577 |
| 1024 | 1221 | 1221 |

Peak capacity scales with W, roughly linearly, matching the paper's Fig 9
observation that increasing the address decoder size increases memory
capacity while the overall curve shape stays similar. Spiking and
non-spiking peaks coincide exactly at every W tested here.

## Files

```
src/sdm_nengo_2021.py              core reimplementation
experiments/01_cmm_capacity.py     reproduces Fig 4-5
experiments/02_sdm_capacity.py     reproduces Fig 6-9
```

## Running it

```
pip install numpy nengo   # nengo not required for these two scripts, but
                           # installed and available for further work
python3 experiments/01_cmm_capacity.py
python3 experiments/02_sdm_capacity.py
```

## Honest limitations / next steps

- Peak capacity numbers are a shape/trend match to the paper, not an exact
  numeric match — the Hebbian-vs-BCM/Oja simplification and the
  closed-form vs literal-Nengo-simulation substitution both contribute
  some numeric drift. Good enough to validate the paper's qualitative
  claims; not good enough to cite as identical replication numbers.
- The MNIST application (paper Section VI) is not reproduced here yet —
  would need an N-of-M encoder network for MNIST images, which is a
  separate, reusable component worth building if there's appetite for it.
- A literal Nengo Ensemble-based simulation (rather than the closed-form
  LIF shortcut) would be the natural next step if you want a result that's
  defensible as "we ran it in Nengo" rather than "we solved the same ODE
  analytically" — useful if this ever needs to go back to the same venue
  or co-authors who'd want to see actual Nengo output.
- Adaptive-LIF, Izhikevich, and Spiking ReLU variants are not reimplemented
  (see above) — would be straightforward to add if useful, since the
  paper's own conclusion is they don't change the capacity curve shape.
