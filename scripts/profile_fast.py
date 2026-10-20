"""Profil jalur panas FastLearner: pembagian waktu per fase + throughput.

  python scripts/profile_fast.py 32 3000            # d, jumlah byte
  python scripts/profile_fast.py 32 3000 --stop     # dengan loss stop aktif

Catatan: fase bersarang (_resolve ⊃ _flush ⊃ flush_rec; _chain ⊃ _stack_step ⊃ step_fwd;
_consume ⊃ slow_update), jadi persentase tidak dijumlahkan ke 100%.
"""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tmt2.model import Config, TMT2
from tmt2.fast_learner import FastLearner
from tmt2.learner import Stream, TrainOpts
from tmt2 import fastpath as fp
from tmt2.corpus import load

d = int(sys.argv[1]) if len(sys.argv) > 1 else 32
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
use_stop = "--stop" in sys.argv

acc = {}


def wrap(obj, name):
    f = getattr(obj, name)

    def g(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            acc[name] = acc.get(name, 0.0) + time.perf_counter() - t
    setattr(obj, name, g)


for n in ("_chain", "_stack_step", "_resolve", "_flush", "_state_cap", "_vic",
          "_consume", "slow_update", "_add", "_farr", "_add_E"):
    wrap(FastLearner, n)
for n in ("step_fwd", "flush_rec", "state_cap_scatter", "head_fwd", "head_bwd",
          "jepa_bwd", "softmax_grad_target", "ce_val"):
    wrap(fp, n)

data = (load("license") * 4)[:N]
cfg = Config(d=d, n_layers=3, d_mem=max(8, d // 4), lr=3e-2, slow_every=16)
m = TMT2(cfg)
m.logit_scale.data = np.array(np.log(3.0))
ln = FastLearner(m, TrainOpts(seed=0, think_prob=0.0, use_stop=use_stop))
s = Stream(m)
ln.step(s, data[0])                                   # warmup + kompilasi njit
t0 = time.perf_counter()
for b in data[1:]:
    ln.step(s, b)
tot = time.perf_counter() - t0
n = len(data) - 1
print(f"d={d} n={n} use_stop={use_stop}  total={tot:.2f}s  {n/tot:.0f} B/s  ({1e3*tot/n:.2f} ms/byte)")
print("  (bersarang: _resolve ⊃ _flush ⊃ flush_rec; _chain ⊃ _stack_step ⊃ step_fwd; _consume ⊃ slow_update)")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<20}{v:7.3f}s {100*v/tot:5.1f}%  {1e6*v/n:8.0f} us/byte")
print("  sleeps:", ln.stats["sleeps"], " slow_updates:", ln.stats["slow_updates"],
      " kedalaman:", ln.stats["depth"].tolist())
