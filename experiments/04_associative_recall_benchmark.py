"""
Long-distractor associative recall benchmark.

Protocol: store a single key->value association (e.g. cue "ABCDEF" maps to
target "XYZ"), then write D unrelated distractor associations, then probe
with the original cue and check whether the target is still recoverable.

This tests long-term associative recall capacity under interference, not
factual/general recall. It is the kind of test where a fixed-capacity,
content-addressable memory (SDM) might degrade differently from a dense
attention/RNN model that has no explicit slot-based storage.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
import torch
import torch.nn as nn
from sequence_machine_2005 import make_rank_order_codes, decode, KanervaSDM


def sdm_recall_test(vocab_size, num_distractors, addr_dim=2048, addr_N=11,
                     M=256, N=11, seed=0):
    codes = make_rank_order_codes(vocab_size, M, N, seed=seed)
    sdm = KanervaSDM(context_dim=M, addr_dim=addr_dim, input_dim=M, addr_N=addr_N, seed=seed)

    rng = random.Random(seed)
    cue, target = 0, 1
    sdm.write(codes[cue], codes[target])

    used = {cue, target}
    for _ in range(num_distractors):
        a = rng.randrange(vocab_size)
        b = rng.randrange(vocab_size)
        while a in (cue,) :
            a = rng.randrange(vocab_size)
        sdm.write(codes[a], codes[b])

    recalled_vec = sdm.read(codes[cue])
    recalled = decode(recalled_vec, codes)
    return recalled == target


class TinyMLPMemory(nn.Module):
    """A dense baseline 'memory': an MLP trained to map cue embedding ->
    target embedding via gradient descent over all the (cue,target) pairs
    seen so far, retrained from scratch after each new distractor (closest
    fair analogue to 'a dense model holding all associations')."""

    def __init__(self, vocab, dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, vocab))

    def forward(self, x):
        return self.net(self.emb(x))


def dense_recall_test(vocab_size, num_distractors, dim=64, epochs=50, seed=0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cue, target = 0, 1
    pairs = [(cue, target)]
    for _ in range(num_distractors):
        a = rng.randrange(vocab_size)
        b = rng.randrange(vocab_size)
        pairs.append((a, b))

    model = TinyMLPMemory(vocab_size, dim)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    xs = torch.tensor([p[0] for p in pairs])
    ys = torch.tensor([p[1] for p in pairs])
    for _ in range(epochs):
        logits = model(xs)
        loss = loss_fn(logits, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor([cue])).argmax(-1).item()
    return pred == target


def main():
    vocab_size = 50
    distractor_counts = [0, 10, 50, 200, 1000, 5000]

    print("=== SDM (one-shot, no retraining) ===")
    for D in distractor_counts:
        ok = sdm_recall_test(vocab_size, D)
        print(f"  distractors={D:5d}  cue recalled correctly: {ok}")

    print("=== Dense MLP baseline (retrained from scratch each time, 50 epochs) ===")
    for D in distractor_counts:
        ok = dense_recall_test(vocab_size, D)
        print(f"  distractors={D:5d}  cue recalled correctly: {ok}")


if __name__ == "__main__":
    main()
