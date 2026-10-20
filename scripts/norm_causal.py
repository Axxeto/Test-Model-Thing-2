"""U6 — APAKAH LEDAKAN NORMA STATE PENYEBAB KEGAGALAN HAFAL?

Temuan: pada 128 byte acak, norma `|y|` lapisan puncak tumbuh geometris
1,7e5 -> 3,2e7 dalam 60 epoch, tanpa mekanisme decay. Gerbang `a = sigmoid(...)`
saturasi (39% a>0.99) sehingga `h = a*hp + (1-a)*u` jadi INTEGRATOR murni:
loop gain > 1, tanpa kebocoran -> norma tak terbatas.

Uji kausal ini membandingkan beberapa penjinak terhadap baseline:
  (0) baseline                 : seperti biasa
  (1) wd>0 (weight decay)      : menekan |Wa|,|B| tumbuh
  (2) clip kecil               : batasi lonjakan gradien
  (3) tanh pada state          : pembatas keras norma state
  (4) init bias gerbang negatif: paksa a kecil di awal (state = u, bukan integrator)

Metrik: CE beku setelah N epoch + pertumbuhan norma state (log10).
Bila penjinak menurunkan CE beku DAN menekan norma -> ledakan norma kausal.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import random_bytes

LN2 = float(np.log(2))


def state_norm(model, data, ln):
    st = Stream(model, online=False)
    ns = []
    for b in data[:64]:
        ln.step(st, int(b), train=False, think=False)
        e = st.buf.get(st.t)
        ns.append(float(np.linalg.norm(e.chain[-1][-1].y)))
    return float(np.mean(ns))


def run(n=128, vocab=64, L=1, dp=0, se=16, epochs=150, d=32, lr=3e-2, d_mem=8,
        wd=0.0, clip=1.0, gate_bias=None, seed=7, tag=""):
    rng = np.random.default_rng(seed)
    data = random_bytes(n, vocab, seed)
    cfg = Config(d=d, n_layers=L, d_mem=d_mem, head="linear", deep_grad=bool(dp), lr=lr,
                 warmup=200, slow_every=se, clip=clip, wd=wd, replay=False, adapter_rank=0,
                 use_innovation=False, mem_query_x=True, mem_wq_match_k=True, mem_hetero=True,
                 mem_surprise_fixed=True, beta_max=1.0, seed=seed)
    m = TMT2(cfg)
    if gate_bias is not None:
        m.params["L0.ba"].data[:] = gate_bias
    ln = FastLearner(m, TrainOpts(seed=seed, think_prob=0.0, use_stop=False, only_byte_loss=True))
    s = Stream(m)
    t0 = time.time()
    for _ in range(epochs):
        for b in data:
            ln.step(s, int(b))
    fz = float(ln.eval_stream([int(b) for b in data]).mean() / LN2)
    sn = state_norm(m, data, ln)
    wa = float(np.linalg.norm(m.params["L0.Wa"].data))
    return dict(fz=fz, sn=sn, logsn=np.log10(sn + 1e-12), wa=wa, secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--L", type=int, default=1)
    a = ap.parse_args()

    trials = [
        ("0 baseline            ", dict()),
        ("1 wd=1e-4             ", dict(wd=1e-4)),
        ("2 wd=1e-3             ", dict(wd=1e-3)),
        ("3 wd=1e-2             ", dict(wd=1e-2)),
        ("4 clip=0.1            ", dict(clip=0.1)),
        ("5 gate_bias=-2        ", dict(gate_bias=np.full(32, -2.0))),
        ("6 gate_bias=-4        ", dict(gate_bias=np.full(32, -4.0))),
        ("7 wd=1e-3 + bias=-2   ", dict(wd=1e-3, gate_bias=np.full(32, -2.0))),
    ]
    print(f"n={a.n} byte ACAK V=64, L={a.L}, {a.epochs} epoch")
    print(f"{'varian':<24}{'beku':>9}{'log10|y|':>10}{'|Wa|':>9}{'detik':>7}")
    print("-" * 60)
    for tag, kw in trials:
        r = run(n=a.n, L=a.L, epochs=a.epochs, **kw)
        print(f"{tag:<24}{r['fz']:>9.4f}{r['logsn']:>10.2f}{r['wa']:>9.2f}{r['secs']:>7.0f}",
              flush=True)
    print("\nBaca: bila 'beku' turun BERSAMAAN dengan log10|y| turun -> ledakan norma kausal.")


if __name__ == "__main__":
    main()
