"""Sweep orde prior n-gram: mengukur KONTRIBUSI MARGINAL SSM di atas lantai n-gram orde-N.

Untuk setiap orde dan seed diukur dua angka pada token yang sama:
  1. LANTAI  : W_out dibekukan nol, byte_bias nol, HANYA ng_lam dilatih -> murni prior + skala.
  2. TMT2    : model penuh, ng_lam dipelajari.
Selisihnya (2 - 1) adalah kontribusi marginal SSM. Kalau tetap ~0 pada orde tinggi, memori
SSM tidak menambah apa pun pada rezim data ini.

  python scripts/ngram_order_sweep.py --data license --orders 1,2,3,4 --seeds 3
  python scripts/ngram_order_sweep.py --data wikitext --n 200000 --bpe --vocab 4096 --orders 1,2,3
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import load
from tmt2 import tokenizer as tk


def prep(a):
    raw = load(a.data)[:a.n]
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if a.bpe:
        cut = int(len(raw) * a.tok_frac)
        pref = "" if a.tok_frac >= 1.0 else f"_pref{int(a.tok_frac*100)}"
        cache = os.path.join(root, "data", f"bpe_{a.data}_{a.vocab}{pref}.json")
        tok = tk.get(raw if a.tok_frac >= 1.0 else raw[:cut], a.vocab, cache)
        ids, blens, canon = tk.encode(tok, raw)
        return ids, blens, a.vocab
    return (np.frombuffer(raw, np.uint8).astype(np.int64),
            np.ones(len(raw), np.int64), 256)


def run(ids, blens, vocab, order, seed, d, lr, warmup, slow_every, floor):
    cfg = Config(d=d, n_layers=3, vocab=vocab, d_mem=max(8, d // 4), adapter_rank=0,
                 slow_every=slow_every, lr=lr, warmup=warmup, seed=seed,
                 head="linear", deep_grad=True, ngram_learn=True, ngram_lam0=1.0)
    m = TMT2(cfg)
    if floor:                                   # W_out & byte_bias nol -> logits = lam*logprior
        m.W_out.data[:] = 0.0
        m.byte_bias.data[:] = 0.0
    opts = dict(seed=seed, think_prob=0.0, only_byte_loss=True, ngram_lam=1.0,
                ngram_order=order)
    if floor:
        opts["only_params"] = ("ng_lam",)
    ln = FastLearner(m, TrainOpts(**opts))
    s = Stream(m)
    bits = np.full(len(ids), np.nan)
    for i, b in enumerate(ids):
        c = ln.step(s, int(b))
        if c is not None:
            bits[i] = c / np.log(2)
    ok = ~np.isnan(bits)
    return float(bits[ok].sum()) / float(blens[ok].sum()), float(m.ng_lam.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="license")
    ap.add_argument("--n", type=int, default=46507)
    ap.add_argument("--bpe", action="store_true")
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--tok_frac", type=float, default=0.5)
    ap.add_argument("--orders", default="1,2,3,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--slow_every", type=int, default=16)
    a = ap.parse_args()

    ids, blens, vocab = prep(a)
    print(f"data={a.data} n={a.n} bpe={a.bpe} vocab={vocab} token={len(ids)} "
          f"({int(blens.sum())} byte)")
    print(f"{'orde':>5}{'lantai':>10}{'TMT2':>10}{'selisih':>10}{'lam akhir':>11}   (rata-rata "
          f"{a.seeds} seed)")
    for order in [int(x) for x in a.orders.split(",")]:
        fl, fu, lams = [], [], []
        for seed in range(a.seeds):
            b1, _ = run(ids, blens, vocab, order, seed, a.d, a.lr, a.warmup, a.slow_every, True)
            b2, lam = run(ids, blens, vocab, order, seed, a.d, a.lr, a.warmup, a.slow_every, False)
            fl.append(b1)
            fu.append(b2)
            lams.append(lam)
        print(f"{order:>5}{np.mean(fl):>10.3f}{np.mean(fu):>10.3f}"
              f"{np.mean(fl)-np.mean(fu):>10.3f}{np.mean(lams):>11.4f}")
    print()
    print("'selisih' > 0 berarti SSM menambah sesuatu di atas lantai n-gram orde tersebut.")


if __name__ == "__main__":
    main()
