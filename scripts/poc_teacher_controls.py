"""Kontrol untuk PoC guru eksternal (mencegah artefak)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.corpus import load
from tmt2.stat_core import StatCore

raw = np.frombuffer(load("wikitext"), dtype=np.uint8)
tr, te = raw[:200000], raw[200000:208000]
SIG = 8.0


def noisy(d, seed):
    rng = np.random.default_rng(seed)
    t = np.empty((len(d), 8))
    for bi in range(8):
        bit = ((d >> (7 - bi)) & 1).astype(np.float64)
        t[:, bi] = np.where(bit > 0.5, 12.0, -12.0)
    return t + rng.normal(0, SIG, size=t.shape)


def randt(d, seed):
    return np.random.default_rng(seed).normal(0, 2.0, size=(len(d), 8))


def mk():
    return StatCore(orders=(0, 1, 2, 3, 4, 5, 6, 8), table_bits=20, wsel_bits=8,
                    match_bits=16, match_fix=True, ctx_fix=True, use_apm=True, mixer_mode=1)


def train(m, tt):
    for i, b in enumerate(tr):
        m.step(int(b), train=True, nn_st=(None if tt is None else tt[i]))


tr_n, te_n = noisy(tr, 1), noisy(te, 2)
tr_r, te_r = randt(tr, 3), randt(te, 4)

m1 = mk(); train(m1, None); print("1 latih TANPA guru, eval guru-derau :", round(float(np.mean(m1.eval_stream(te, nn_stream=te_n))), 4))
m2 = mk(); train(m2, tr_n); print("2 latih guru-derau, eval guru-ACAK :", round(float(np.mean(m2.eval_stream(te, nn_stream=te_r))), 4))
m3 = mk(); train(m3, tr_n); print("3 latih guru-derau, eval TANPA guru:", round(float(np.mean(m3.eval_stream(te))), 4))
m4 = mk(); train(m4, tr_n); print("4 latih guru-derau, eval guru-derau:", round(float(np.mean(m4.eval_stream(te, nn_stream=te_n))), 4))
m5 = mk(); train(m5, tr_r); print("5 latih guru-ACAK,  eval guru-derau:", round(float(np.mean(m5.eval_stream(te, nn_stream=te_n))), 4))
m6 = mk(); train(m6, None); print("6 latih TANPA guru, eval TANPA guru:", round(float(np.mean(m6.eval_stream(te))), 4))
