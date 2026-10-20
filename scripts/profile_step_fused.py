"""Profil rinci step_fused: pisahkan biaya kernel vs scaffolding Python.

Ukur dengan time.perf_counter per segmen pada loop byte nyata (bukan microbench terisolasi),
supaya overhead dispatch & interpreter yang sebenarnya ikut terhitung.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.corpus import load
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2 import fastpath as fp

P = time.perf_counter


def build(d=32, nl=2, V=256, slow_every=10 ** 9):
    cfg = Config.best(d=d, n_layers=nl, vocab=V, d_mem=max(8, d // 4), use_memory=True,
                      head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                      think_max=0, use_innovation=False, seg_len=10 ** 9,
                      slow_every=slow_every)
    m = TMT2(cfg)
    o = TrainOpts(only_byte_loss=True, use_fused_kernel=True, seed=0, use_stop=False)
    L = FastLearner(m, o)
    return cfg, m, L


def main():
    data = np.frombuffer(load("license"), dtype=np.uint8)
    d, V = 32, 256
    cfg, m, L = build(d=d, V=V)
    s = Stream(m, online=True)
    nw = 400
    for i in range(nw):
        L.step_fused(s, int(data[i]) % V, train=True)
    L.gacc, L.touched, L.nacc = {}, set(), 0

    N = 3000
    tot = dict(ce=0.0, glo=0.0, bwd=0.0, x=0.0, fwd=0.0, snap=0.0, prom=0.0, logc=0.0, tail=0.0)
    for i in range(N):
        b = int(data[nw + i]) % V
        t0 = P()
        ce_pre = fp.ce_val(L._f_logits_prev, b) if L._f_has_prev else None
        t1 = P()
        if L._f_has_prev:
            G = L.gacc.get("byte")
            if G is None:
                G = L.gacc["byte"] = np.zeros(L.ntot)
            glo = fp.softmax_grad_target(L._f_logits_prev, b, 1.0)
            t2 = P()
            ox0 = L.o_E + L._f_byte_prev * d
            fp.fused_bwd((L.f_bundle, L.f_st_prev, L.f_loff, L._f_x_prev,
                          glo, L._f_byte_prev, ox0, G, None, L.f_scrb,
                          L.f_act, L.f_aoff))
            t3 = P()
            L.touched.add("E")
            for l in range(L.f_nl):
                p = f"L{l}."
                L.touched.update((p + "Wc", p + "Wq", p + "Wm", p + "Wg1", p + "Wg2",
                                  p + "Wo", p + "Wa", p + "B", p + "ba"))
            L.touched.update(("W_out", "byte_bias", f"P{L.f_nl - 1}_1"))
            t4 = P()
            tot["glo"] += t2 - t1
            tot["bwd"] += t3 - t2
            tot["tail"] += t4 - t3
        else:
            tot["glo"] += 0.0
        x = m.E.data[int(b)].copy()
        t5 = P()
        fp.fused_fwd((L.f_bundle, L.f_st, L.f_loff, x, L.f_out, L.f_scr,
                      L.f_act, L.f_aoff))
        t6 = P()
        L.f_st_prev[:] = L.f_st
        t7 = P()
        fp.fused_promote_mem(L.f_st, d, L.f_dm, L.f_nl)
        t8 = P()
        logits = L.f_out[d:d + V]
        L._f_logits_prev = logits.copy()
        L._f_byte_prev = int(b)
        L._f_x_prev = x
        L._f_has_prev = True
        t9 = P()
        tot["ce"] += t1 - t0
        tot["x"] += t5 - (t4 if L._f_has_prev or True else t5)
        tot["fwd"] += t6 - t5
        tot["snap"] += t7 - t6
        tot["prom"] += t8 - t7
        tot["logc"] += t9 - t8
        tot["allsum"] = tot.get("allsum", 0.0) + (t9 - t0)

    tsum = sum(v for k, v in tot.items() if k != "allsum")
    print(f"N={N} byte   total {tot['allsum']*1e6/N:6.2f} us/byte  ({N/tot['allsum']:.0f} B/s)")
    print(f"  {'segmen':16s} {'us/byte':>8s} {'%':>7s}")
    for k in ("ce", "glo", "bwd", "tail", "x", "fwd", "snap", "prom", "logc"):
        print(f"  {k:16s} {tot[k]*1e6/N:8.2f} {100*tot[k]/tsum:6.1f}%")
    print(f"  {'JUMLAH':16s} {tsum*1e6/N:8.2f}")
    # selisih ke total = overhead loop skrip
    print(f"  selisih (loop)   {(tot['allsum']-tsum)*1e6/N:8.2f}")


if __name__ == "__main__":
    main()
