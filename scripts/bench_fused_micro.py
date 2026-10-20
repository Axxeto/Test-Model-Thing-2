"""Microbench kernel fused (layout AKTUAL): fused_fwd, fused_bwd (stashing aktivasi),
promote, decay, + scaffolding Python (ce_val, softmax_grad_target, touched.update, salin).

Tujuan: angka per-kernel yang bersih (tanpa instrumentasi di dalam loop byte), supaya
terlihat kernel mana yang dominan dan berapa biaya Python nyata.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.corpus import load
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner

P = time.perf_counter


def micro(fn, n):
    fn()  # warm
    t0 = P()
    for _ in range(n):
        fn()
    return (P() - t0) / n * 1e6


def main():
    data = np.frombuffer(load("license"), dtype=np.uint8)
    for d in (32, 64):
        V, nl = 256, 2
        cfg = Config.best(d=d, n_layers=nl, vocab=V, d_mem=max(8, d // 4), use_memory=True,
                          head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                          think_max=0, use_innovation=False, seg_len=10 ** 9,
                          slow_every=10 ** 9)
        m = TMT2(cfg)
        o = TrainOpts(only_byte_loss=True, use_fused_kernel=True, seed=0, use_stop=False)
        L = FastLearner(m, o)
        s = Stream(m, online=True)
        for i in range(400):
            L.step_fused(s, int(data[i]) % V, train=True)
        L.gacc.setdefault("byte", np.zeros(L.ntot))
        cef = __import__("tmt2.fastpath", fromlist=["x"])
        L.glo_d = cef.softmax_grad_target(L.f_out[d:d + V], 3, 1.0)
        N = 4000
        # --- kernel terpisah
        t_fwd = micro(lambda: cef.fused_fwd((L.f_bundle, L.f_st, L.f_loff, L._f_x_prev,
                                            L.f_out, L.f_scr, L.f_act, L.f_aoff,
                                            L._f_logits_prev, L._f_byte_prev)), N)
        t_bwd = micro(lambda: cef.fused_bwd((L.f_bundle, L.f_st_prev, L.f_loff, L._f_x_prev,
                                            L._f_logits_prev, L._f_byte_prev, L.o_E, L.gacc["byte"],
                                            None, L.f_scrb, L.f_act, L.f_aoff, 0)), N)
        t_bwd_rc = micro(lambda: cef.fused_bwd((L.f_bundle, L.f_st_prev, L.f_loff, L._f_x_prev,
                                               L._f_logits_prev, L._f_byte_prev, L.o_E,
                                               L.gacc["byte"], None, L.f_scrb, L.f_act,
                                               L.f_aoff, 1)), N)
        t_prom = micro(lambda: cef.fused_promote_mem(L.f_st, L.f_d, L.f_dm, L.f_nl), N)
        t_dec = micro(lambda: cef.fused_decay_traces(L.f_st, L.f_d, L.f_dm, L.f_nl, 0.5), N)
        t_ce = micro(lambda: cef.ce_val(L.f_out[d:d + V], 3), N)
        t_glo = micro(lambda: cef.softmax_grad_target(L.f_out[d:d + V], 3, 1.0), N)
        t_snap = micro(lambda: L.f_st_prev.__setitem__(slice(None), L.f_st), N)
        t_logc = micro(lambda: L.f_out[d:d + V].copy(), N)
        # touched.update: union frozenset terprakomputasi (kini jalan sebenarnya)
        t_tch = micro(lambda: L.touched.__ior__(L._f_touched), N)
        # step_fused utuh
        t_all = micro(lambda: L.step_fused(s, 42, train=True), N)
        print(f"d={d}:  step_fused {t_all:6.1f} us ({1e6/t_all:.0f} B/s)")
        print(f"   fwd {t_fwd:6.2f}  bwd {t_bwd:6.2f} (rc {t_bwd_rc:6.2f})  prom {t_prom:5.2f}  "
              f"dec {t_dec:5.2f}  snap {t_snap:5.2f}  logc {t_logc:5.2f}  tch {t_tch:5.2f}")
        kern = t_fwd + t_bwd + t_prom + t_dec + t_snap + t_logc + t_tch
        print(f"   jumlah kernel {kern:6.1f} us; Python sisa {t_all-kern:6.1f} us")
        print(f"   (ce {t_ce:5.2f} & glo {t_glo:5.2f} sudah DIFUSI ke kernel -> tidak dibayar)")


if __name__ == "__main__":
    main()
