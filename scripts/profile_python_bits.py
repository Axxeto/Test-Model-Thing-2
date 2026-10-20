"""Bedah biaya Python per-byte di FastLearner.step (bukan njit).

Membungkus operasi-operasi Python (bukan kernel) dengan timer:
  - alokasi & dict churn di _stack_step (LayerState/FRec/pool.get)
  - _add / gacc get-or-create / np.zeros(ntot)
  - _flush bookkeeping (set, sorted, touched.update)
  - _consume bookkeeping
  - slow_update pecahan (norm, adam, decay, touched loop)

Target: menemukan 298 us/byte Python yang tersisa setelah njit ~66 us.

  python scripts/profile_python_bits.py --d 32 --n 1500
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import TrainOpts, Stream
from tmt2 import fast_learner as FL
from tmt2.fast_learner import FastLearner
from tmt2.corpus import load

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=32)
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--corpus", default="license")
a = ap.parse_args()

acc = {}
def wrap(name, label=None):
    f = getattr(FastLearner, name); label = label or name
    def g(self, *ar, **kw):
        t = time.perf_counter()
        try:
            return f(self, *ar, **kw)
        finally:
            acc[label] = acc.get(label, 0.0) + time.perf_counter() - t
    setattr(FastLearner, name, g)

for n_ in ("_add", "_flush", "_resolve", "_consume", "slow_update", "_stack_step", "step"):
    wrap(n_)

raw = load(a.corpus)
ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) if isinstance(raw, bytes) else raw
cfg = Config.best(vocab=256, d=a.d, n_layers=a.layers, d_mem=max(8, a.d // 4),
                  slow_every=8, seg_len=256, replay=False, sleep_every=10 ** 9)
cfg.mixer_k = 0
m = TMT2(cfg)
fl = FastLearner(m, TrainOpts(seed=0, think_prob=0.0, only_byte_loss=True, use_stop=False))
s = Stream(m, online=True)
for b in ids[:300]:
    fl.step(s, int(b), train=True, think=False)
acc.clear()
s = Stream(m, online=True)
t0 = time.perf_counter()
for b in ids[300:300 + a.n]:
    fl.step(s, int(b), train=True, think=False)
tot = time.perf_counter() - t0
print(f"d={a.d} layers={a.layers} n={a.n}: total={tot:.2f}s  {a.n/tot:.0f} B/s  ({1e6*tot/a.n:.1f} us/byte)")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<14}{v:8.3f}s {100*v/tot:5.1f}%  {1e6*v/a.n:8.1f} us/byte")
n_slow = m.cfg.slow_every
print(f"\n  slow_update dipanggil ~{a.n//n_slow} kali; per-call = {1e6*acc.get('slow_update',0)/max(1,a.n//n_slow):.1f} us")
print(f"  panggilan njit/byte = 9 -> biaya njit aktual ~66.4 us/byte (mb_kernel.py)")
