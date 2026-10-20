"""TUGAS 1: isolasi total -- latih HANYA byte_bias (opsional + logit_scale).

Semua parameter lain (SSM, encoder E, kepala JEPA, adaptor) DIBEKUKAN pada nilai inisialisasi.
Tujuannya: apakah loss + optimizer sanggup menurunkan CE(softmax(byte_bias)) sampai entropi
unigram (4,54 bpb pada korpus lisensi)? Jika tidak, ada bug di loss/optimizer -- bukan kapasitas.

  python scripts/bias_only.py --n 5000 --params byte_bias
  python scripts/bias_only.py --n 5000 --params byte_bias,logit_scale
  python scripts/bias_only.py --n 1000 --epochs 10 --params byte_bias   # uji overfit
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import load


def ce_of_bias(bb, truth):
    """CE dalam bit bila prediksi HANYA dari softmax(byte_bias)."""
    z = bb - bb.max()
    p = np.exp(z) / np.exp(z).sum()
    return float(-np.log2(np.maximum(p[truth], 1e-300)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--report", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=1, help=">1 hanya berguna untuk uji overfit")
    ap.add_argument("--params", default="byte_bias")
    ap.add_argument("--out_lr", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    only = tuple(x for x in a.params.split(",") if x)
    base = load("license")[:a.n]
    data = base * a.epochs
    truth = np.frombuffer(data, np.uint8)
    tc = np.bincount(np.frombuffer(base, np.uint8), minlength=256)
    ce_uni = float(-np.log2(np.maximum(tc / tc.sum(), 1e-300)[truth]).mean())

    cfg = Config(d=a.d, n_layers=3, d_mem=max(8, a.d // 4), lr=3e-2, warmup=20,
                 slow_every=16, seed=a.seed)
    model = TMT2(cfg)
    model.logit_scale.data = np.array(np.log(3.0))
    bb0 = model.byte_bias.data.copy()
    print(f"trainable      = {only}   (semua lain dibekukan)")
    print(f"inisialisasi   : byte_bias min/max/mean = {bb0.min():.4f}/{bb0.max():.4f}/{bb0.mean():.4f}"
          f"   logit_scale(exp) = {float(np.exp(model.logit_scale.data)):.4f}")
    print(f"target         : CE unigram = {ce_uni:.3f} bpb   |   uniform ln256 = {np.log2(256):.3f}")
    print(f"data           : {len(data)} byte ({a.epochs} epoch x {a.n})\n", flush=True)

    ln = FastLearner(model, TrainOpts(seed=a.seed, think_prob=0.0, use_stop=False,
                                      only_params=only, out_lr_mult=a.out_lr, debug_out=True))
    s = Stream(model)
    t0 = time.time()
    for i, b in enumerate(data):
        ln.step(s, b)
        if (i + 1) % a.report == 0:
            g = ln.dbg[-1][3] if ln.dbg else {}
            print(f"  byte {i+1:>6}  CE(bias)={ce_of_bias(model.byte_bias.data, truth):.3f}  "
                  f"logit_scale={float(np.exp(model.logit_scale.data)):.4f}  " +
                  "  ".join(f"||g_{k}||={v:.2e}" for k, v in g.items()), flush=True)
    dt = time.time() - t0

    bb = model.byte_bias.data
    print(f"\nAKHIR: CE(softmax(byte_bias)) = {ce_of_bias(bb, truth):.3f} bpb   "
          f"(unigram {ce_uni:.3f}, uniform {np.log2(256):.3f})")
    print(f"       logit_scale(exp) = {float(np.exp(model.logit_scale.data)):.4f}")
    print(f"       byte_bias berubah: max|Δ| = {np.abs(bb - bb0).max():.4f}")
    gs = {k: [] for k in ln.dbg_names}
    for _, _, _, g in ln.dbg:
        for k, v in g.items():
            gs[k].append(v)
    print("       rata-rata ||g|| = " + "  ".join(f"{k}={np.mean(v):.3e}" for k, v in gs.items() if v))
    print(f"       korelasi byte_bias vs log(unigram) = "
          f"{np.corrcoef(bb, np.log(tc / tc.sum() + 1e-12))[0,1]:.3f}")
    print(f"       waktu {dt:.1f}s")


if __name__ == "__main__":
    main()
