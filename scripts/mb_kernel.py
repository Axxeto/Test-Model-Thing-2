"""Mikro-benchmark kernel njit SATUAN + orkestrasi Python per-byte.

Tujuan: pisahkan biaya nyata
  (a) kernel njit (step_fwd, flush_rec, head_fwd_lin, head_bwd_lin_acc, softmax_grad_target, ce_val)
  (b) orkestrasi Python FastLearner (alokasi, dict, LayerState, dsb)

Jalankan:  python scripts/mb_kernel.py --d 32 --layers 2 --reps 200
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2, l2n_np, ln_np
from tmt2.core import LayerState
from tmt2 import fastpath as fp

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=32)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--dm", type=int, default=None)
a = ap.parse_args()

d = a.d
cfg = Config.best(vocab=256, d=d, n_layers=a.layers, d_mem=a.dm or max(8, d // 4), slow_every=8)
m = TMT2(cfg)
dm = cfg.d_mem
ed = cfg.expand * d
r = cfg.adapter_rank
print(f"d={d} dm={dm} ed={ed} r={r} V={cfg.vocab}")

scr = np.zeros(fp.scratch_size(d, dm, ed, r))
scrb = np.zeros(fp.scratch_size_bwd(d, dm, ed, r))
scrmw = np.zeros(fp.scratch_size_mw(d, dm))


def timeit(label, fn, n):
    fn()                      # jit compile
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n
    print(f"  {label:<28}{1e6*dt:9.2f} us")
    return dt


# ---- state lapisan (LayerState butuh (d, dm))
st = LayerState(d, dm)
L0 = m.layers[0]
x = m.E.data[7].copy()
e = np.zeros(d)


def k_step_fwd():
    fp.step_fwd(x, st.h, st.S["Wa"], st.S["B"], st.S["K"], st.S["ba"],
                L0.Wa.data, L0.ba.data, L0.B.data, L0.K.data, e, False,
                st.M, L0.Wc.data, L0.Wq.data, L0.Wm.data, L0.Wg1.data, L0.Wg2.data,
                L0.Wo.data, L0.Af.data, L0.Bf.data,
                cfg.use_memory, cfg.adapter_rank > 0,
                L0.Wk.data, L0.Wv.data, st.kp, 0.0, cfg.beta_max, cfg.mem_hetero,
                cfg.mem_surprise_fixed, cfg.mem_query_x,
                st.h, st.S["Wa"], st.S["B"], st.S["K"], st.S["ba"], st.h, st.M, st.kp,
                st.h, st.h, scr)


print("\n-- kernel SATUAN (per panggilan) --")
t_step = timeit("step_fwd (1 lapisan)", k_step_fwd, a.reps)

gy = np.zeros(d)
g = np.zeros(1024 * d)
o_gam = np.zeros(dm); o_q = np.zeros(dm); o_gx = np.zeros(d)
gAf = np.zeros((d, r)); gBf = np.zeros((r, d))
lo = {k: 0 for k in ("Wc", "Wq", "Wm", "Wg1", "Wg2", "Wo", "Wa", "B", "ba", "K")}


def k_flush():
    fp.flush_rec(st.h, x, st.M, gy, L0.Wc.data, L0.Wq.data, L0.Wm.data,
                 L0.Wg1.data, L0.Wg2.data, L0.Wo.data, L0.Af.data, L0.Bf.data,
                 st.S["Wa"], st.S["B"], st.S["K"], st.S["ba"],
                 L0.Wa.data, L0.B.data, st.h, st.h,
                 False, cfg.use_memory, cfg.adapter_rank > 0, cfg.mem_query_x,
                 cfg.deep_grad_gate, g, gAf, gBf,
                 lo["Wc"], lo["Wq"], lo["Wm"], lo["Wg1"], lo["Wg2"], lo["Wo"],
                 lo["Wa"], lo["B"], lo["ba"], lo["K"], -1,
                 o_gam, o_q, o_gx, scrb)


t_flush = timeit("flush_rec (1 lapisan)", k_flush, a.reps)

p = np.zeros(d); W_out = m.W_out.data; bb = m.byte_bias.data
glo = np.zeros(cfg.vocab)
t_head_f = timeit("head_fwd_lin", lambda: fp.head_fwd_lin(p, W_out, bb), a.reps)
t_head_b = timeit("head_bwd_lin_acc", lambda: fp.head_bwd_lin_acc(p, W_out, glo, 1.0,
                                                                 np.eye(d), p, g, 0), a.reps)
t_smg = timeit("softmax_grad_target", lambda: fp.softmax_grad_target(p[:cfg.vocab] if cfg.vocab <= d else np.zeros(cfg.vocab), 3, 1.0), a.reps)
t_ce = timeit("ce_val", lambda: fp.ce_val(np.zeros(cfg.vocab), 3), a.reps)

print("\n-- ringkasan per-byte (1 lapisan) --")
per_layer = t_step + t_flush
print(f"  step_fwd+flush_rec /lapisan      {1e6*per_layer:9.2f} us")
print(f"  kepala (fwd+bwd+smg+ce)          {1e6*(t_head_f+t_head_b+t_smg+t_ce):9.2f} us")
print(f"  TOTAL njit (2 lapisan + kepala)  {1e6*(2*per_layer+t_head_f+t_head_b+t_smg+t_ce):9.2f} us")
