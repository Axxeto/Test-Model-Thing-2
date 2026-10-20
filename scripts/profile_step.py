"""Profil ringan: pembagian waktu per fase Learner.step (timer terbungkus, overhead rendah).

  python scripts/profile_step.py 32 1500      # argumen: d, jumlah byte

Catatan: fase bersarang (_chain ⊃ stack_step ⊃ layer_step; _resolve ⊃ backward_task ⊃ rtrl_flush;
_consume ⊃ slow_update; sleep ⊃ step penuh), jadi persentase tidak dijumlahkan ke 100%.
"""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tmt2.model import Config, TMT2
from tmt2 import learner as LM, core as CM
from tmt2.learner import Learner, Stream, TrainOpts
from tmt2.corpus import load
raw = load("license")                    # lintas-platform (data/licenses/ atau /usr/share/...)
d = int(sys.argv[1]); N = int(sys.argv[2])
acc = {}
def wrap(obj, name, label=None):
    f = getattr(obj, name); label = label or name
    def g(*a, **k):
        t = time.perf_counter()
        try: return f(*a, **k)
        finally: acc[label] = acc.get(label, 0.0) + time.perf_counter() - t
    setattr(obj, name, g)
for n in ("_chain","_resolve","_state_cap","_vic","_consume","slow_update","sleep"):
    wrap(Learner, n)
# bagian dalam _chain / _resolve
wrap(LM, "stack_step"); wrap(LM, "backward_task"); wrap(LM, "cross_entropy")
wrap(CM, "layer_step"); wrap(CM, "rtrl_flush")
cfg = Config(d=d, n_layers=3, d_mem=max(8, d//4), lr=3e-2, slow_every=8)
m = TMT2(cfg); m.logit_scale.data = np.array(np.log(3.0))
ln = Learner(m, TrainOpts()); s = Stream(m)
t0 = time.perf_counter()
for b in raw[:N]: ln.step(s, b)
tot = time.perf_counter() - t0
print(f"d={d} n={N}  total={tot:.2f}s  {N/tot:.0f} B/s  ({1e3*tot/N:.2f} ms/byte)  params={m.n_params()}")
print("  (bersarang: _chain ⊃ stack_step ⊃ layer_step; _resolve ⊃ backward_task ⊃ rtrl_flush; sleep ⊃ step penuh)")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<14}{v:7.2f}s {100*v/tot:5.1f}%  {1e6*v/N:8.0f} µs/byte")
print("  sleeps:", ln.stats["sleeps"], " slow_updates:", ln.stats["slow_updates"], " kedalaman:", ln.stats["depth"].tolist())
