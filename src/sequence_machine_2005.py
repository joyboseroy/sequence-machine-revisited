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
    """Separate context neural layer (Elman-style), section VII: new context
    is a fixed nonlinear (here: random linear + clamp) function of old
    context (scaled by sensitivity lambda) and new input."""

    def __init__(self, M: int, lam: float = 0.2, seed: int = 0):
        self.M = M
        self.lam = lam
        g = torch.Generator().manual_seed(seed + 1)
        self.scramble = torch.rand(M, M, generator=g) * 0.1 + torch.eye(M) * 0.9

    def init_context(self) -> torch.Tensor:
        return torch.zeros(self.M)

    def step(self, old_context: torch.Tensor, input_code: torch.Tensor) -> torch.Tensor:
        scrambled = self.scramble @ old_context
        return torch.tanh(self.lam * scrambled + input_code)


class CombinedContext(ContextModel):
    """The 'combined model' (section VIII / Fig.4): old context is
    deterministically scrambled, scaled by x<1, the input code is added in
    expanded form, and the K largest components are kept as the new
    K-of-M context (K >> N). This is the model the original papers found to
    outperform both pure shift-register and pure context-layer schemes."""

    def __init__(self, ctx_dim: int, input_dim: int, K: int, x: float = 0.3,
                 seed: int = 0):
        self.ctx_dim = ctx_dim
        self.input_dim = input_dim
        self.K = K
        self.x = x
        g = torch.Generator().manual_seed(seed + 2)
        # deterministic scramble: fixed random projection ctx_dim -> ctx_dim
        self.scramble = torch.rand(ctx_dim, ctx_dim, generator=g)
        # expand input_dim -> ctx_dim
        self.expand = torch.rand(ctx_dim, input_dim, generator=g)

    def init_context(self) -> torch.Tensor:
        return torch.zeros(self.ctx_dim)

    def step(self, old_context: torch.Tensor, input_code: torch.Tensor) -> torch.Tensor:
        scrambled = self.scramble @ old_context
        expanded_input = self.expand @ input_code
        combined = self.x * scrambled + expanded_input
        topk = torch.topk(combined, self.K).indices
        new_ctx = torch.zeros(self.ctx_dim)
        new_ctx[topk] = combined[topk]
        return new_ctx


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

    def run_sequence(self, seq, learn_pass=True):
        """Run a sequence once. Returns list of predictions made *before*
        each symbol was learned (on-line, single-pass prediction)."""
        self.reset()
        preds = []
        for sym in seq:
            preds.append(self.step(sym))
        return preds
