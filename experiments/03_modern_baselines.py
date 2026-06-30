"""
Modern baselines on the SAME task and SAME protocol as the 2005 sequence
machine: single online pass over a symbol sequence (predicting the next
symbol at every step), alphabet size 15, then test recall.

Note this is an intentionally unfair-looking setup for gradient-based
models: a single forward+backward pass per symbol, no epochs, no large
pretraining corpus. That is exactly the point: the 2005 machine was
explicitly built for one-shot on-line learning, while Transformers/LSTMs
are normally trained over many epochs on much more data. This experiment
asks what happens when both are forced into the SDM's regime: one shot,
on-line, tiny data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
import torch
import torch.nn as nn


def random_sequence(alphabet_size, length, seed):
    rng = random.Random(seed)
    return [rng.randrange(alphabet_size) for _ in range(length)]


class TinyLSTM(nn.Module):
    def __init__(self, vocab, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, state=None):
        e = self.emb(x)
        h, state = self.lstm(e, state)
        return self.out(h), state


class TinyTransformer(nn.Module):
    """Causal mini-Transformer, single layer, run over the whole prefix at
    each step (since Transformers have no native recurrent state)."""

    def __init__(self, vocab, dim=64, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=128,
                                            batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.out = nn.Linear(dim, vocab)

    def forward(self, x):
        T = x.shape[1]
        pos_ids = torch.arange(T).unsqueeze(0)
        h = self.emb(x) + self.pos(pos_ids)
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.out(h)


def online_lstm_errors(seq, vocab, lr=0.05, hidden=64, seed=0):
    torch.manual_seed(seed)
    model = TinyLSTM(vocab, hidden)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # online learning pass: at each step, predict next symbol, then update
    # weights from the true label (one gradient step per symbol, single pass)
    state = None
    x_prev = None
    model.train()
    for sym in seq:
        x_t = torch.tensor([[sym]])
        if x_prev is not None:
            logits, state = model(x_prev, state)
            loss = loss_fn(logits[:, -1, :], x_t[:, 0])
            opt.zero_grad()
            loss.backward()
            opt.step()
            state = (state[0].detach(), state[1].detach())
        x_prev = x_t

    # test pass: re-run from scratch (fresh hidden state), single pass,
    # count next-symbol prediction errors using the now-trained weights
    model.eval()
    state = None
    errors = 0
    with torch.no_grad():
        for i in range(len(seq) - 1):
            x_t = torch.tensor([[seq[i]]])
            logits, state = model(x_t, state)
            pred = logits[:, -1, :].argmax(-1).item()
            if pred != seq[i + 1]:
                errors += 1
    return errors


def online_transformer_errors(seq, vocab, lr=0.05, dim=64, seed=0):
    torch.manual_seed(seed)
    model = TinyTransformer(vocab, dim)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    prefix = []
    for sym in seq:
        if len(prefix) >= 1:
            x = torch.tensor([prefix])
            target = torch.tensor([sym])
            logits = model(x)
            loss = loss_fn(logits[:, -1, :], target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        prefix.append(sym)

    model.eval()
    errors = 0
    with torch.no_grad():
        prefix = []
        for i in range(len(seq) - 1):
            prefix.append(seq[i])
            x = torch.tensor([prefix])
            logits = model(x)
            pred = logits[:, -1, :].argmax(-1).item()
            if pred != seq[i + 1]:
                errors += 1
    return errors


def main():
    alphabet_size = 15
    lengths = [30, 60, 90]
    for L in lengths:
        seq = random_sequence(alphabet_size, L, seed=L)
        lstm_err = online_lstm_errors(seq, alphabet_size)
        tfm_err = online_transformer_errors(seq, alphabet_size)
        print(f"len={L:3d}  LSTM_errors={lstm_err:3d}  Transformer_errors={tfm_err:3d}")


if __name__ == "__main__":
    main()
