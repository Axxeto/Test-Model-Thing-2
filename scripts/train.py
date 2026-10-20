"""Latih TMT2 online pada byte stream; laporkan bits/byte PREKUENSIAL dan bandingkan dengan
baseline n-gram adaptif (order 0/1/2) pada data yang sama.

  python scripts/train.py --data file.txt --n 20000 --d 32 --layers 3 --out ckpt.npz
"""
import argparse, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Learner, Stream, TrainOpts
from tmt2.corpus import load
from tmt2.fast_learner import FastLearner


def ngram_bits(data, order):
    """bits/byte prekuensial n-gram adaptif dengan back-off (interpolasi, gaya PPM sederhana)."""
    counts = [dict() for _ in range(order + 1)]
    out = []
    for i, b in enumerate(data):
        p = 1.0 / 256
        for o in range(order + 1):
            if i < o:
                break
            ctx = bytes(data[i - o:i])
            c = counts[o].get(ctx)
            if c is None:
                c = counts[o][ctx] = [np.zeros(256), 0.0]
            p = (c[0][b] + 2.0 * p) / (c[1] + 2.0)
        out.append(-np.log2(p))
        for o in range(order + 1):
            if i < o:
                break
            c = counts[o][bytes(data[i - o:i])]
            c[0][b] += 1; c[1] += 1
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None); ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--d", type=int, default=32); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--think_prob", type=float, default=0.25)
    ap.add_argument("--init_scale", type=float, default=3.0, help="logit_scale awal (config asli: 10)")
    ap.add_argument("--lr", type=float, default=3e-2, help="lr AdamW jalur lambat (aliran pendek butuh lr besar)")
    ap.add_argument("--slow_every", type=int, default=8, help="periode pembaruan jalur lambat (byte)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true",
                    help="pakai FastLearner (jalur panas @njit + arena datar); butuh numba")
    ap.add_argument("--no_stop", action="store_true", help="matikan loss stop (fokus byte+JEPA)")
    ap.add_argument("--out", default=None); ap.add_argument("--report", type=int, default=2000)
    a = ap.parse_args()
    if a.data:
        raw = open(a.data, "rb").read()
    else:
        raw = load("license")            # lintas-platform (data/licenses/ atau /usr/share/...)
    data = raw[:a.n]
    cfg = Config(d=a.d, n_layers=a.layers, d_mem=max(8, a.d // 4), lr=a.lr, warmup=20,
                 slow_every=a.slow_every, seed=a.seed)
    model = TMT2(cfg)
    model.logit_scale.data = np.array(np.log(a.init_scale))
    K = FastLearner if a.fast else Learner
    ln = K(model, TrainOpts(think_prob=a.think_prob, seed=a.seed, use_stop=not a.no_stop))
    print(f"learner={'FastLearner(@njit)' if a.fast else 'Learner(autograd)'} "
          f"use_stop={not a.no_stop}", flush=True)
    print(f"parameter={model.n_params()} byte={len(data)} d={cfg.d} lapisan={cfg.n_layers}", flush=True)
    s, bits, t0 = Stream(model), [], time.time()
    for i, b in enumerate(data):
        c = ln.step(s, b)
        if c is not None:
            bits.append(c / np.log(2))
        if (i + 1) % a.report == 0:
            print(f"  byte {i+1:>7}  bpb({a.report} terakhir)={np.mean(bits[-a.report:]):.3f}  "
                  f"{(i+1)/(time.time()-t0):.0f} B/s  unc={ {k: round(v,2) for k,v in ln.unc.items()} } "
                  f"kedalaman={ln.stats['depth'].tolist()} sleep={ln.stats['sleeps']}", flush=True)
    bits = np.array(bits)
    print("\n== bits/byte prekuensial (semakin kecil semakin baik) ==")
    third = len(bits) // 3
    rows = [("TMT2 (online)", bits)] + [(f"n-gram order {o}", ngram_bits(data, o)[1:]) for o in (0, 1, 2)]
    print(f"{'model':<18}{'semua':>8}{'sepertiga akhir':>18}")
    for n, b in rows:
        print(f"{n:<18}{b.mean():>8.3f}{b[-third:].mean():>18.3f}")
    if a.out:
        ln.save(a.out); print("checkpoint ->", a.out)
    print("\ncontoh generasi:", ln.generate(data[:32], 80, temp=0.7, seed=1))


if __name__ == "__main__":
    main()
