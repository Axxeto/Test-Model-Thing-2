"""Bedah slow_update (994 us/8 byte = 124 us/byte) -- hambatan #1.

Mengukur tiap tahap secara langsung (bukan via cProfile yang salah atribusi):
  1. prepare gb (divide, weight, pcgrad)
  2. norm + clip
  3. AdamW in-place (6 op array ntot)
  4. loop touched -> theta update (per parameter)
  5. unc + ema_update + target_dict + decay_traces + clear

  python scripts/profile_slow_update.py --d 32 --n 3200
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
ap.add_argument("--n", type=int, default=3200)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--corpus", default="license")
a = ap.parse_args()

raw = load(a.corpus)
ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) if isinstance(raw, bytes) else raw
cfg = Config.best(vocab=256, d=a.d, n_layers=a.layers, d_mem=max(8, a.d // 4),
                  slow_every=8, seg_len=256, replay=False, sleep_every=10 ** 9)
cfg.mixer_k = 0
m = TMT2(cfg)
fl = FastLearner(m, TrainOpts(seed=0, think_prob=0.0, only_byte_loss=True, use_stop=False))
s = Stream(m, online=True)
for b in ids[:400]:
    fl.step(s, int(b), train=True, think=False)

print(f"ntot={fl.ntot}  n_touched={len(fl.touched)}  slow_every={cfg.slow_every}")
print(f"touched names: {sorted(fl.touched)[:8]} ...")

N = 200
# 1. prepare
def p_prepare():
    K = float(fl.nacc)
    wt = [k for k in FL.W_TASKS if k in fl.gacc]
    gb = fl.gbuf
    if len(wt) == 1:
        np.divide(fl.gacc[wt[0]], K, out=gb)
        gb *= fl._weight(wt[0])
    return gb

def p_norm():
    return float(np.linalg.norm(fl.gbuf))

def p_adam():
    cfgx = cfg
    gb, t1, t2 = fl.gbuf, fl.t1, fl.t2
    b1, b2 = cfgx.betas
    lr = cfgx.lr
    fl.adam_m *= b1
    np.multiply(gb, 1.0 - b1, out=t2); fl.adam_m += t2
    np.multiply(gb, gb, out=t1)
    fl.adam_v *= b2
    np.multiply(t1, 1.0 - b2, out=t2); fl.adam_v += t2
    t2[:] = fl.adam_v; t2 /= (1.0 - b2 ** fl.t_adam); np.sqrt(t2, out=t2); t2 += 1e-8
    t1[:] = fl.adam_m; t1 /= (1.0 - b1 ** fl.t_adam); np.divide(t1, t2, out=t1)

def p_touched():
    lr = cfg.lr
    dcy = 1.0 - lr * cfg.wd
    for name in fl.touched:
        o = fl.offs[name][0]; sz = fl.sizes[name]
        v = fl.theta[o:o + sz]
        if fl.mat_of[name]:
            v *= dcy
        np.multiply(fl.t1[o:o + sz], lr * fl.lrmul_of[name], out=fl.t2[o:o + sz])
        v -= fl.t2[o:o + sz]

def p_tail():
    fl.m.ema_update(cfg.ema)
    fl.tdict = fl.m.target_dict()
    for st in {id(x): x for x in (s, fl.live) if x is not None}.values():
        st.st.decay_traces(cfg.s_decay)

def timeit(label, fn, n=N):
    fn(); t0 = time.perf_counter()
    for _ in range(n): fn()
    dt = (time.perf_counter() - t0) / n
    print(f"  {label:<26}{1e6*dt:9.1f} us   ({1e6*dt*cfg.slow_every/1:6.1f} us/byte eq)")
    return dt

print("\n-- per tahap (per panggilan slow_update) --")
timeit("prepare (divide+weight)", p_prepare)
timeit("norm(gb)", p_norm)
timeit("adam in-place", p_adam)
timeit("touched -> theta loop", p_touched)
timeit("tail (ema+dict+decay)", p_tail)
timeit("FULL slow_update", lambda: fl.slow_update(s))

# perbandingan norm
gb = fl.gbuf
def n_linalg(): return float(np.linalg.norm(gb))
def n_dot():    return float(np.sqrt(gb @ gb))
def n_mul():    return float(np.sqrt((gb * gb).sum()))
print("\n-- alternatif norm (N=%d) --" % gb.size)
timeit("np.linalg.norm", n_linalg, 500)
timeit("sqrt(gb @ gb)", n_dot, 500)
timeit("sqrt((gb*gb).sum())", n_mul, 500)
