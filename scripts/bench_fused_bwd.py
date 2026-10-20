"""Benchmark backward: fused_bwd (1 panggilan) vs jalur lama (head_bwd + n_layers flush_rec)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2 import fastpath as fp

NAMES = ("Wa", "ba", "B", "K", "Wc", "Wq", "Wm", "Wg1", "Wg2", "Wo", "Af", "Bf", "Wk", "Wv")

for nl, d, dm, V in ((2, 32, 8, 256), (2, 64, 16, 256)):
    cfg = Config.best(d=d, n_layers=nl, vocab=V, d_mem=dm, use_memory=True,
                      head="linear", deep_grad=True)
    m = TMT2(cfg)
    ed, r = cfg.expand * d, cfg.adapter_rank
    ssz = fp.state_size_per_layer(d, dm)
    st = np.zeros(nl * ssz)
    ntot = 300_000
    loff = np.zeros((nl, 12), dtype=np.int64)
    for l in range(nl):
        for i in range(12):
            loff[l, i] = (l * 12 + i) * 800
    oWout, oBias, oP = ntot - V * d - V - d * d, ntot - V - d * d, ntot - d * d

    par = []
    for l in range(nl):
        L = m.layers[l]
        par.append(tuple(getattr(getattr(L, nm_, None), "data", np.zeros((d, dm))) for nm_ in NAMES))
    bundle = (d, dm, ed, r, V, nl, 1, 0, int(cfg.mem_query_x), 0, 1, 0, cfg.beta_max,
              tuple(par), m.W_out.data, m.byte_bias.data, m.P[(nl - 1, 1)].data,
              0, 1, oWout, oBias, oP)
    scr_f = np.zeros(fp.scratch_size(d, dm, ed, r) + 2 * d)
    scrb = np.zeros(fp.scratch_size_bwd(d, dm, ed, r) + 8 * d + V + 3 * d)
    rng = np.random.default_rng(0)
    xin = rng.standard_normal(d)
    out = np.zeros(d + V + 2 * nl * d)
    fp.fused_fwd(bundle, st, loff, xin, out, scr_f)
    glo = fp.softmax_grad_target(out[d:d + V], 3, 1.0)
    G = np.zeros(ntot)
    scrr = np.zeros(fp.scratch_size_bwd(d, dm, ed, r))

    def old_bwd():
        Ltop = nl - 1
        base = Ltop * ssz
        ytop = st[base + d + 3 * d * d + d:base + d + 3 * d * d + 2 * d]
        gy, gP, gv = fp.head_bwd_lin_acc(ytop @ m.P[(Ltop, 1)].data, m.W_out.data, glo, 1.0,
                                         m.P[(Ltop, 1)].data, ytop, G, oWout)
        G[oBias:oBias + V] += glo
        Pfull = ytop @ m.P[(Ltop, 1)].data
        nrm = np.sqrt((Pfull * Pfull).sum() + 1e-8)
        pv = Pfull / nrm
        g_s = m.W_out.data.T @ glo
        gp = (g_s - pv * (g_s @ pv)) / nrm
        G[oP:oP + d * d] += np.outer(ytop, gp).reshape(-1)
        gy_cur = gy.copy()
        for l in range(nl - 1, -1, -1):
            base = l * ssz
            h = st[base:base + d]
            SWa = st[base + d:base + d + d * d].reshape(d, d)
            SB = st[base + d + d * d:base + d + 2 * d * d].reshape(d, d)
            SK = st[base + d + 2 * d * d:base + d + 3 * d * d].reshape(d, d)
            Sba = st[base + d + 3 * d * d:base + d + 3 * d * d + d]
            M = st[base + d + 3 * d * d + 2 * d:base + d + 3 * d * d + 2 * d + dm * dm].reshape(dm, dm)
            aa = st[base + d + 3 * d * d + 3 * d + dm * dm:base + d + 3 * d * d + 4 * d + dm * dm]
            ccs = st[base + d + 3 * d * d + 4 * d + dm * dm:base + d + 3 * d * d + 5 * d + dm * dm]
            x = xin if l == 0 else st[(l - 1) * ssz + d + 3 * d * d + d:(l - 1) * ssz + d + 3 * d * d + 2 * d]
            L = m.layers[l]; lo = loff[l]
            o_gam = np.zeros(dm); o_q = np.zeros(dm); o_gx = np.zeros(d)
            gAf = np.zeros((d, r)); gBf = np.zeros((r, d))
            fp.flush_rec(h, x, M, gy_cur, L.Wc.data, L.Wq.data, L.Wm.data, L.Wg1.data,
                         L.Wg2.data, L.Wo.data, L.Af.data, L.Bf.data, SWa, SB, SK, Sba,
                         L.Wa.data, L.B.data, aa, ccs, False, cfg.use_memory, r > 0,
                         cfg.mem_query_x, 1, G, gAf, gBf,
                         lo[4], lo[5], lo[6], lo[7], lo[8], lo[9], lo[0], lo[2], lo[1], lo[3],
                         -1, o_gam, o_q, o_gx, scrr)
            if l > 0:
                gy_cur = o_gx + gy_cur

    def new_bwd():
        fp.fused_bwd(bundle, st, loff, xin, glo, 3, -1, G, None, scrb)

    old_bwd(); new_bwd()
    N = 3000
    t0 = time.perf_counter()
    for _ in range(N):
        old_bwd()
    dt_old = (time.perf_counter() - t0) / N
    t0 = time.perf_counter()
    for _ in range(N):
        new_bwd()
    dt_new = (time.perf_counter() - t0) / N
    print(f"nl={nl} d={d} V={V}:  lama {1e6*dt_old:7.2f} us  |  fused {1e6*dt_new:7.2f} us  "
          f"|  speedup {dt_old/dt_new:.2f}x")
