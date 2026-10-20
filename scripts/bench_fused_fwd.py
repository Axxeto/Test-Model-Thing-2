"""Benchmark: fused_fwd (1 panggilan) vs jalur lama (n_layers x step_fwd + head).

Target: membuktikan pengurangan dispatch. Jalur lama = n_layers panggilan 42-arg + 1 kepala.
Fused = 1 panggilan 17-arg. Us/byte diukur dari loop murni (state in-place).
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2 import fastpath as fp

for nl, d, dm, V in ((2, 32, 8, 256), (2, 64, 16, 256)):
    cfg = Config.best(d=d, n_layers=nl, vocab=V, d_mem=dm, use_memory=True,
                      head="linear", deep_grad=True)
    m = TMT2(cfg)
    ed = cfg.expand * d
    r = cfg.adapter_rank
    ssz = fp.state_size_per_layer(d, dm)
    st = np.zeros(nl * ssz)
    loff = np.zeros((nl, 12), dtype=np.int64)
    names = ("Wa", "ba", "B", "K", "Wc", "Wq", "Wm", "Wg1", "Wg2", "Wo", "Af", "Bf", "Wk", "Wv")
    par = []
    for l in range(nl):
        L = m.layers[l]
        row = [getattr(getattr(L, nm_, None), "data", getattr(L, nm_, np.zeros((d, dm)))) for nm_ in names]
        par.append(tuple(row))
    par = tuple(par)
    bundle = (d, dm, ed, r, V, nl, 1, 0, int(cfg.mem_query_x), 0, 1, 0,
              cfg.beta_max, par, m.W_out.data, m.byte_bias.data, m.P[(nl - 1, 1)].data)
    scr_f = np.zeros(fp.scratch_size(d, dm, ed, r) + 2 * d)
    out = np.zeros(d + V + 2 * nl * d)
    xin = np.random.default_rng(0).standard_normal(d)

    # ---- lama: n_layers step_fwd + head
    scr_o = np.zeros(fp.scratch_size(d, dm, ed, r))
    hs = [np.zeros(d) for _ in range(nl)]
    SWs = [np.zeros((d, d)) for _ in range(nl)]
    SBs = [np.zeros((d, d)) for _ in range(nl)]
    SKs = [np.zeros((d, d)) for _ in range(nl)]
    Sbas = [np.zeros(d) for _ in range(nl)]
    Ms = [np.zeros((dm, dm)) for _ in range(nl)]
    kps = [np.zeros(d) for _ in range(nl)]

    def old_path():
        cur = xin
        for l in range(nl):
            L = m.layers[l]
            fp.step_fwd(cur, hs[l], SWs[l], SBs[l], SKs[l], Sbas[l],
                        L.Wa.data, L.ba.data, L.B.data, L.K.data, np.zeros(d), False,
                        Ms[l], L.Wc.data, L.Wq.data, L.Wm.data, L.Wg1.data, L.Wg2.data,
                        L.Wo.data, L.Af.data, L.Bf.data, cfg.use_memory, False,
                        L.Wk, L.Wv, kps[l], 0.0, cfg.beta_max, False, True, cfg.mem_query_x,
                        hs[l], SWs[l], SBs[l], SKs[l], Sbas[l],
                        scr_o[:d], Ms[l], kps[l], scr_o[:d], scr_o[:d], scr_o)
            cur = scr_o[:d]
        return fp.head_fwd_lin(cur @ m.P[(nl - 1, 1)].data, m.W_out.data, m.byte_bias.data)

    def new_path():
        fp.fused_fwd(bundle, st, loff, xin, out, scr_f)
        return out[d:d + V]

    # warmup (compile)
    old_path(); new_path()
    N = 4000
    t0 = time.perf_counter()
    for _ in range(N):
        old_path()
    dt_old = (time.perf_counter() - t0) / N
    t0 = time.perf_counter()
    for _ in range(N):
        new_path()
    dt_new = (time.perf_counter() - t0) / N
    print(f"nl={nl} d={d} V={V}:  lama {1e6*dt_old:7.2f} us  |  fused {1e6*dt_new:7.2f} us  "
          f"|  speedup {dt_old/dt_new:.2f}x  ({1/dt_old:6.0f} -> {1/dt_new:6.0f} B/s maju saja)")
