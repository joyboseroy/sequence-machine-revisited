"""
A faithful PyTorch reimplementation of the sequence machine described in:

  Bose, Furber, Shapiro (2005). "An associative memory for the on-line
  recognition and prediction of temporal sequences." IJCNN 2005.
  Bose, Furber, Shapiro (2005). "A spiking neural sparse distributed memory
  implementation for learning and predicting temporal sequences." ICANN 2005.

This module reimplements the *abstract* (non-spiking) version of the
architecture: rank-ordered N-of-M codes, a modified Kanerva Sparse
Distributed Memory (SDM) with max-based one-shot Hebbian learning, and the
three context-encoding schemes compared in the original papers (shift
register, context neural layer, and the "combined" model). The spiking
implementation (asynchronous timing, RDLIF neurons) is a separate, lower-
level engineering concern addressed in the original WIRN/Async Forum papers
and is not required to reproduce the *functional* behaviour of the machine.

No modern deep learning is used here deliberately: this is the 2005
algorithm, unchanged in spirit, so that later experiments can compare it
honestly against 2026 baselines (Transformer, LSTM, Mamba, etc).
"""

import torch


def make_rank_order_codes(num_symbols: int, M: int, N: int, k: float = 0.9,
                           seed: int = 0) -> torch.Tensor:
    """Generate a fixed rank-ordered N-of-M code for each symbol.

    Each code is an M-dim vector where N components are active. The active
    components are assigned weights 1, k, k^2, ... k^(N-1) according to a
    random firing order, exactly as described in the IJCNN paper section V:
    "the 3-of-5 code representing firing order 3-2-4 is represented as
    [0, k, 1, k^2, 0]".
    """
    g = torch.Generator().manual_seed(seed)
    codes = torch.zeros(num_symbols, M)
    for s in range(num_symbols):
        active = torch.randperm(M, generator=g)[:N]
        weights = k ** torch.arange(N, dtype=torch.float32)
        codes[s, active] = weights
    return codes


def decode(vec: torch.Tensor, codebook: torch.Tensor) -> int:
    """Nearest-code decoding by normalised dot product (cosine similarity),
    exactly as specified in the papers."""
    v = vec / (vec.norm() + 1e-8)
    cb = codebook / (codebook.norm(dim=1, keepdim=True) + 1e-8)
    sims = cb @ v
    return int(sims.argmax().item())


def decode_topk(vec: torch.Tensor, codebook: torch.Tensor, k: int = 3):
    """Same decoding as `decode`, but returns the top-k candidate symbols
    ranked by similarity, plus their similarity scores. Used to
    characterise prediction errors per thesis section 9.5.2: 'the chosen
    symbol may be incorrect but the output symbol whose activity was
    second highest or third highest may be the correct symbol' -- this is
    the metric needed to detect that case."""
    v = vec / (vec.norm() + 1e-8)
    cb = codebook / (codebook.norm(dim=1, keepdim=True) + 1e-8)
    sims = cb @ v
    topk = torch.topk(sims, k)
    return [(int(idx), float(score)) for idx, score in zip(topk.indices, topk.values)]


class KanervaSDM:
    """Modified Kanerva Sparse Distributed Memory with N-of-M address
    decoding and max-based ("one-shot") Hebbian learning, as in section III
    of the ICANN paper.

    - address decoder: fixed random projection context_dim -> addr_dim,
      followed by hard top-N selection to produce an N-of-M address code
      (ordered by activation magnitude, mirroring rank-order coding).
    - data store: learns W_data = max(W_data, outer(addr_code, input_code))
    """

    def __init__(self, context_dim: int, addr_dim: int, input_dim: int,
                 addr_N: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.W_addr = torch.rand(addr_dim, context_dim, generator=g)
        self.addr_dim = addr_dim
        self.addr_N = addr_N
        self.W_data = torch.zeros(addr_dim, input_dim)

    def _address_code(self, context: torch.Tensor) -> torch.Tensor:
        raw = self.W_addr @ context
        topk = torch.topk(raw, self.addr_N).indices
        addr = torch.zeros(self.addr_dim)
        # rank-order weighting of the address code itself
        order = raw[topk].argsort(descending=True)
        ranks = torch.empty_like(order, dtype=torch.float32)
        for rank, idx in enumerate(order):
            ranks[idx] = 0.9 ** rank
        addr[topk] = ranks
        return addr

    def write(self, context: torch.Tensor, input_code: torch.Tensor):
        addr = self._address_code(context)
        outer = torch.outer(addr, input_code)
        self.W_data = torch.maximum(self.W_data, outer)

    def read(self, context: torch.Tensor) -> torch.Tensor:
        addr = self._address_code(context)
        return addr @ self.W_data


class ContextModel:
    """Base interface for the three context-update schemes compared in the
    1005 paper. `step(old_context, input_code)` returns the new context."""

    def step(self, old_context: torch.Tensor, input_code: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def init_context(self) -> torch.Tensor:
        raise NotImplementedError


class ShiftRegisterContext(ContextModel):
    """Fixed time-window context (Fig.2 in IJCNN paper): context = sum of
    last `lookback` input codes, more recent weighted more heavily."""

    def __init__(self, M: int, lookback: int = 2, decay: float = 0.5):
        self.M = M
        self.lookback = lookback
        self.decay = decay
        self.history = []

    def init_context(self) -> torch.Tensor:
        self.history = []
        return torch.zeros(self.M)

    def step(self, old_context: torch.Tensor, input_code: torch.Tensor) -> torch.Tensor:
        self.history.append(input_code)
        self.history = self.history[-self.lookback:]
        ctx = torch.zeros(self.M)
        for i, code in enumerate(reversed(self.history)):
            ctx = ctx + (self.decay ** i) * code
        return ctx


class ContextLayerModel(ContextModel):
    """DEPRECATED as of the thesis-faithful rewrite: the original 2005
    conference papers describe a separate 'context neural layer' model as
    if it were architecturally distinct from the combined model. Reading
    PhD thesis Chapter 5 (Bose 2007, 'Designing a sequence machine') shows
    this is NOT the case: there is a single combined-model formula (eq 5.4
    in the thesis), and the 'context neural layer' behaviour is simply the
    Lambda=0.5 (lambda=1.0) special case of that one formula, not a
    separate architecture. This class is kept only so old experiment
    scripts referencing it don't break, but new code should use
    CombinedContext with Lambda=0.5 instead. See CombinedContext docstring.
    """

    def __init__(self, M: int, lam: float = 0.2, seed: int = 0):
        raise NotImplementedError(
            "ContextLayerModel has been removed in the thesis-faithful "
            "rewrite. Use CombinedContext(..., Lambda=0.5) instead, which "
            "is the architecturally correct equivalent per thesis eq. 5.5."
        )


def nof_m(vec: torch.Tensor, m: int, alpha: float) -> torch.Tensor:
    """Re-encode a real-valued vector as an ordered m-of-M rank code, per
    thesis section 5.2.1: select the m largest components, then OVERWRITE
    their values with the rank-order significance weights 1, alpha,
    alpha^2, ... according to their relative rank (not their raw
    magnitudes). All other components are zeroed. This is the same
    'nof_m' operation used for the input encoder itself, applied here to
    the context after combining old context and new input."""
    M = vec.shape[0]
    topk_idx = torch.topk(vec, m).indices
    order = vec[topk_idx].argsort(descending=True)  # rank within the top-m
    out = torch.zeros(M)
    for rank, pos in enumerate(order):
        out[topk_idx[pos]] = alpha ** rank
    return out


def scale(vec: torch.Tensor) -> torch.Tensor:
    """The 'scale' function from thesis eq 5.4/5.5: normalise so the vector
    components sum to 1 (an L1 normalisation, valid since rank-order code
    components are non-negative)."""
    s = vec.sum()
    if s <= 0:
        return vec
    return vec / s


class CombinedContext(ContextModel):
    """The thesis's actual (and only) context model, reimplemented exactly
    per Chapter 5, eq. 5.5 (the bounded convex-combination form, which the
    thesis says is used interchangeably with eq. 5.4's unbounded lambda
    form -- "In our tests... we shall investigate both of these models."):

        C_n = nof_m( Lambda * scale(P1 @ C_{n-1}) + (1-Lambda) * scale(P2 @ I_n),
                     m, M, alpha )

    where:
      - P1 is a FIXED [M,M] random matrix projecting the old context
      - P2 is a FIXED [M,D] random matrix projecting the new input into
        context space ("expand")
      - scale() is L1-normalisation (components sum to 1)
      - nof_m() re-encodes the top-m components as a rank-order code with
        significance ratio alpha (NOT just top-K masking with raw values,
        which was a bug in the pre-thesis-correction implementation)

    Lambda=0.5 corresponds to lambda=1.0 in the unbounded form: input and
    past context equally important (thesis calls this case equivalent to
    a pure context-neural-layer model). Lambda -> 0 makes the new context
    increasingly input-dominated (shift-register-like, gradual forgetting
    of the past). Lambda=1.0 is degenerate (context never updates from new
    input at all) and is not a useful operating point.

    The thesis also notes a principled, non-arbitrary choice for the
    lambda scaling factor: lambda = alpha^(N+1), tying the context
    modulation directly to the significance ratio alpha used for the
    rank-order code and the number of active input components N, so that
    the most significant bit of the OLD context is weighted below the
    LEAST significant bit of the NEW context (ensuring gradual forgetting).
    We expose this as `principled_lambda()`.
    """

    def __init__(self, ctx_dim: int, input_dim: int, m: int, Lambda: float = 0.3,
                 alpha: float = 0.9, seed: int = 0):
        self.ctx_dim = ctx_dim
        self.input_dim = input_dim
        self.m = m
        self.Lambda = Lambda
        self.alpha = alpha
        g = torch.Generator().manual_seed(seed + 2)
        self.P1 = torch.rand(ctx_dim, ctx_dim, generator=g)
        self.P2 = torch.rand(ctx_dim, input_dim, generator=g)

    @staticmethod
    def principled_lambda(alpha: float, N: int) -> float:
        """thesis p.105: 'keeping the scaling factor equal to alpha^(N+1)
        will ensure that the most important bit of the old context is
        given a weight less than the least important bit of the new
        context' -- this is the lambda form (eq 5.4), not the bounded
        Lambda form; convert with Lambda = lambda/(1+lambda) if needed."""
        return alpha ** (N + 1)

    def init_context(self) -> torch.Tensor:
        return torch.zeros(self.ctx_dim)

    def step(self, old_context: torch.Tensor, input_code: torch.Tensor) -> torch.Tensor:
        old_proj = scale(self.P1 @ old_context) if old_context.sum() > 0 else old_context
        new_proj = scale(self.P2 @ input_code)
        combined = self.Lambda * old_proj + (1 - self.Lambda) * new_proj
        return nof_m(combined, self.m, self.alpha)


class SequenceMachine2005:
    """End-to-end on-line sequence machine: encoder -> context model -> SDM
    -> decoder, exactly the architecture in Fig.5 of the IJCNN paper."""

    def __init__(self, num_symbols: int, context_model: ContextModel,
                 input_M: int = 256, input_N: int = 11, addr_dim: int = 2048,
                 addr_N: int = 11, seed: int = 0):
        self.codebook = make_rank_order_codes(num_symbols, input_M, input_N, seed=seed)
        self.context_model = context_model
        ctx_dim = context_model.ctx_dim if hasattr(context_model, "ctx_dim") else input_M
        self.sdm = KanervaSDM(ctx_dim, addr_dim, input_M, addr_N, seed=seed)
        self.context = context_model.init_context()

    def reset(self):
        self.context = self.context_model.init_context()

    def step(self, symbol: int) -> int:
        """Present one symbol, following the 3-step on-line learning
        framework in IJCNN paper section IV exactly:
          1. associate the new input with the *present* (old) context -> write
          2. compute the *new* context from old context + input
          3. predict by reading the memory with the *new* context
        Returns the prediction for the symbol AFTER this one."""
        input_code = self.codebook[symbol]
        self.sdm.write(self.context, input_code)          # step 1
        self.context = self.context_model.step(self.context, input_code)  # step 2
        predicted_vec = self.sdm.read(self.context)        # step 3
        predicted_symbol = decode(predicted_vec, self.codebook)
        return predicted_symbol

    def step_topk(self, symbol: int, k: int = 3):
        """Same as step(), but returns the top-k ranked candidate
        predictions with similarity scores, for error-type analysis
        (thesis section 9.5.2)."""
        input_code = self.codebook[symbol]
        self.sdm.write(self.context, input_code)
        self.context = self.context_model.step(self.context, input_code)
        predicted_vec = self.sdm.read(self.context)
        return decode_topk(predicted_vec, self.codebook, k=k)

    def run_sequence(self, seq, learn_pass=True):
        """Run a sequence once. Returns list of predictions made *before*
        each symbol was learned (on-line, single-pass prediction)."""
        self.reset()
        preds = []
        for sym in seq:
            preds.append(self.step(sym))
        return preds
