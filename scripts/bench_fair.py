"""Benchmark ADIL FastLearner: njit-AdamW vs numpy-AdamW, model+warmup identik per pembanding.

PELAJARAN METODOLOGI (lihat catatan kecepatan): benchmark yang memakai ULANG satu model antar-rep
menghasilkan angka B/s terlalu TINGGI karena pada rep-2+ state sudah konvergen dan `touched`
sudah penuh, sehingga `slow_update` terukur lebih murah dari kondisi awal. Skrip ini membangun
model BARU + warmup identik untuk SETIAP pembanding dan setiap rep -> angka apple-to-apple.

  python scripts/bench_fair.py --d 32 64 128
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import TrainOpts, Stream
from tmt2.fast_learner import FastLearner
from tmt2.corpus import load

ap = argparse.ArgumentParser()
ap.add_argument("--dims", default="32,64,128")
ap.add_argument("--warm", type=int, default=500)
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--corpus", default="license")
a = ap.parse_args()

raw = load(a.corpus)
ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) if isinstance(raw, bytes) else raw


def run(use_njit, d, warm, n, reps):
    dm = max(8, d // 4)
    best = 1e9
    for _ in range(reps):
        cfg = Config.best(vocab=256, d=d, n_layers=2, d_mem=dm, slow_every=8,
                          seg_len=256, replay=False, sleep_every=10 ** 9)
        cfg.mixer_k = 0
        m = TMT2(cfg)
        fl = FastLearner(m, TrainOpts(seed=0, think_prob=0.0, only_byte_loss=True,
                                      use_stop=False, use_njit_adam=use_njit))
        s = Stream(m, online=True)
        for b in ids[:warm]:
            fl.step(s, int(b), train=True, think=False)
        s = Stream(m, online=True)
        t0 = time.perf_counter()
        for b in ids[warm:warm + n]:
            fl.step(s, int(b), train=True, think=False)
        best = min(best, time.perf_counter() - t0)
    return n / best


print(f"korpus={a.corpus} warm={a.warm} n={a.n} reps={a.reps} (model+warmup baru per rep)")
for d in [int(x) for x in a.dims.split(",")]:
    nj = run(True, d, a.warm, a.n, a.reps)
    np_ = run(False, d, a.warm, a.n, a.reps)
    print(f"  d={d:3d}: njit={nj:7.0f} B/s ({1e6/nj:6.1f} us/byte)  "
          f"numpy={np_:7.0f} B/s ({1e6/np_:6.1f} us/byte)  speedup={nj/np_:.2f}x")
