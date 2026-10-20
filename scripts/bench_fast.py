"""Benchmark jalur PANAS (FastLearner) — baseline + rincian biaya.

  python scripts/bench_fast.py --d 32 --n 4000

Melaporkan B/s untuk mode terbaik (head=linear, deep_grad, only_byte) pada data lisensi nyata.

PERINGATAN METODOLOGI: skrip ini MEMAKAI ULANG satu model antar-rep. Pada rep-2+ state sudah
konvergen dan `touched` penuh, sehingga angka B/s cenderung LEBIH TINGGI dari kondisi awal.
Untuk perbandingan apple-to-apple (mis. use_njit_adam True vs False), pakai
`scripts/bench_fair.py` yang membangun model+warmup baru per pembanding.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import TrainOpts, Stream
from tmt2.fast_learner import FastLearner
from tmt2.corpus import load


def bench(d=32, n=4000, layers=2, d_mem=None, mixer=0, warm=300, corpus="license", reps=1):
    raw = load(corpus)
    if isinstance(raw, bytes):
        ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    else:
        ids = raw
    ids = ids[:n + warm]
    cfg = Config.best(vocab=256, d=d, n_layers=layers, d_mem=d_mem or max(8, d // 4),
                      slow_every=8, seg_len=256, replay=False, sleep_every=10 ** 9)
    if mixer == 0:
        cfg.mixer_k = 0
    m = TMT2(cfg)
    fl = FastLearner(m, TrainOpts(seed=0, think_prob=0.0, only_byte_loss=True, use_stop=False))
    s = Stream(m, online=True)
    for b in ids[:warm]:                         # pemanasan (jit compile + state konvergen)
        fl.step(s, int(b), train=True, think=False)
    best = 1e9
    for _ in range(reps):
        s2 = Stream(m, online=True)
        t0 = time.perf_counter()
        for b in ids[warm:warm + n]:
            fl.step(s2, int(b), train=True, think=False)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    bps = n / best
    print(f"d={d} layers={layers} d_mem={cfg.d_mem} mixer_k={cfg.mixer_k} "
          f"n={n}: {bps:8.0f} B/s  ({1e6*best/n:7.1f} us/byte)")
    return bps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--mixer", type=int, default=0)
    ap.add_argument("--corpus", default="license")
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    bench(a.d, a.n, a.layers, mixer=a.mixer, corpus=a.corpus, reps=a.reps)
