"""Profil fase FastLearner.step (jalur panas nyata). Timer dibungkus per-metode.

  python scripts/profile_fast_phases.py --d 32 --n 2000

Fase bersarang (step > _chain > _stack_step; step > _resolve > _consume), jadi persen
TIDAK dijumlahkan ke 100%. Tujuan: menemukan di mana waktu Python (bukan njit) habis.
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
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--mixer", type=int, default=0)
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

for n_ in ("_chain", "_stack_step", "_resolve", "_consume", "_flush", "slow_update", "step"):
    if hasattr(FastLearner, n_):
        wrap(n_)

raw = load(a.corpus)
ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) if isinstance(raw, bytes) else raw
cfg = Config.best(vocab=256, d=a.d, n_layers=a.layers, d_mem=max(8, a.d // 4),
                  slow_every=8, seg_len=256, replay=False, sleep_every=10 ** 9)
if a.mixer == 0:
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
print(f"d={a.d} layers={a.layers} mixer_k={cfg.mixer_k} n={a.n}: total={tot:.2f}s  "
      f"{a.n/tot:.0f} B/s  ({1e6*tot/a.n:.1f} us/byte)")
print("  (bersarang: step > _chain > _stack_step; step > _resolve > _consume/_flush)")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<14}{v:8.3f}s {100*v/tot:5.1f}%  {1e6*v/a.n:8.1f} us/byte")
