"""Bandingkan flush_rec vs fused_bwd pada STATE YANG SAMA (dari f_st_prev LF), byte-2 backward.

Kita jalankan referensi flush_rec dengan input identik (diambil dari arena f_st_prev) dan
bandingkan scatter gradien per-parameter blok.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2 import fastpath as fp

D, DM, ED, R, V, NL = 16, 8, 32, 0, 64, 2


def make(use_memory):
    cfg = Config.best(d=D, n_layers=NL, vocab=V, d_mem=DM, use_memory=use_memory,
                      head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                      think_max=0, use_innovation=False)
    m = TMT2(cfg)
    o = TrainOpts(only_byte_loss=True, use_fused_kernel=True, seed=0, use_stop=False)
    return cfg, m, FastLearner(m, o)


def main():
    data = [3, 17, 42]
    cfg, m, LF = make(True)
    s = Stream(m, online=True)
    for b in data:
        LF.step_fused(s, b, train=True)

    # ambil state yang dipakai backward terakhir: f_st_prev (state sebelum byte terakhir)
    st = LF.f_st_prev
    xin = LF._f_x_prev
    logits_prev = LF._f_logits_prev      # logits byte-(t-1)
    byte_prev = LF._f_byte_prev
    glo = fp.softmax_grad_target(logits_prev, int(data[-1]), 1.0)

    # ---- (A) fused_bwd -> arena G1
    G1 = np.zeros(LF.ntot)
    ox0 = LF.o_E + byte_prev * D
    fp.fused_bwd(LF.f_bundle, st, LF.f_loff, xin, glo, byte_prev, ox0, G1, None, LF.f_scrb)

    # ---- (B) referensi: head_bwd_lin_acc + flush_rec per lapisan (top->bottom)
    G2 = np.zeros(LF.ntot)
    ssz = LF.f_ssz
    so = fp.state_offsets(NL, D, DM)[1]
    W_out = LF.m.W_out.data; byte_bias = LF.m.byte_bias.data
    Ptop = LF.m.P[(NL - 1, 1)].data
    top_base = (NL - 1) * ssz
    ytop = st[top_base + so["y"]:top_base + so["y"] + D]
    g_y, gP, g_bb = fp.head_bwd_lin_acc(ytop @ Ptop, W_out, glo, 1.0, Ptop, ytop, G2, LF.offs["W_out"][0])
    G2[LF.offs["byte_bias"][0]:LF.offs["byte_bias"][0] + V] += g_bb
    oP = LF.offs[f"P{NL-1}_1"][0]
    G2[oP:oP + D * D] += gP.reshape(-1)

    scrb = LF.f_scrb
    gy = g_y.copy()
    for l in range(NL - 1, -1, -1):
        base = l * ssz
        h = st[base + so["h"]:base + so["h"] + D]
        M = st[base + so["M"]:base + so["M"] + DM * DM].reshape(DM, DM)
        aa = st[base + so["a"]:base + so["a"] + D]
        ccs = st[base + so["cc"]:base + so["cc"] + D]
        SWa = st[base + so["SWa"]:base + so["SWa"] + D * D].reshape(D, D)
        SB = st[base + so["SB"]:base + so["SB"] + D * D].reshape(D, D)
        SK = st[base + so["SK"]:base + so["SK"] + D * D].reshape(D, D)
        Sba = st[base + so["Sba"]:base + so["Sba"] + D]
        x = xin if l == 0 else st[(l - 1) * ssz + so["y"]:(l - 1) * ssz + so["y"] + D]
        L = m.layers[l]
        lo = LF.loff[l]
        ox = -1
        if l == 0:
            ox = LF.o_E + byte_prev * D
        gAf = LF._zero_r0; gBf = LF._zero_r0
        o_gam = np.zeros(DM); o_q = np.zeros(DM); o_gx = np.zeros(D)
        fp.flush_rec(h, x, M, gy, L.Wc.data, L.Wq.data, L.Wm.data, L.Wg1.data, L.Wg2.data,
                     L.Wo.data, L.Af.data, L.Bf.data, SWa, SB, SK, Sba, L.Wa.data, L.B.data,
                     aa, ccs, True, True, False, cfg.mem_query_x, cfg.deep_grad_gate,
                     G2, gAf, gBf, lo["Wc"], lo["Wq"], lo["Wm"], lo["Wg1"], lo["Wg2"], lo["Wo"],
                     lo["Wa"], lo["B"], lo["ba"], lo["K"], ox, o_gam, o_q, o_gx, scrb)
        if l > 0:
            gy = (gy + o_gx) if cfg.deep_grad_gate else gy

    # ---- bandingkan
    print("total max diff:", np.abs(G1 - G2).max())
    for name, (off, shp) in LF.offs.items():
        n = int(np.prod(shp)) if len(shp) else 1
        dv = float(np.abs(G1[off:off + n] - G2[off:off + n]).max())
        if dv > 1e-12:
            print(f"  {name:14s} {dv:.3e}")


if __name__ == "__main__":
    main()
