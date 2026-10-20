"""Uji eksak: kernel fused_fwd vs jalur lama (step_fwd + head_fwd_lin), SATU langkah.

Protocol: bangun model kecil, inisialisasi state dari nol, jalankan SATU langkah maju
lewat kedua jalur dari x yang sama, bandingkan h/y/M/kp/logits bit-demi-bit.

Jangan uji lintasan panjang -- chaos online menggeser hasil walau beda 1e-15.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2, l2n_np
from tmt2 import fastpath as fp


def build(nl=2, d=16, dm=8, mem=True, V=64):
    cfg = Config.best(d=d, n_layers=nl, vocab=V, d_mem=dm, use_memory=mem,
                      head="linear", deep_grad=True)
    m = TMT2(cfg)
    return cfg, m


def ref_step_fwd_layer(m, cfg, l, xin, h, SWa, SB, SK, Sba, M, kp, scr):
    L = m.layers[l]
    d, dm = cfg.d, cfg.d_mem
    o_h = np.zeros(d); o_SWa = np.zeros((d, d)); o_SB = np.zeros((d, d))
    o_SK = np.zeros((d, d)); o_Sba = np.zeros(d); o_y = np.zeros(d)
    o_M = np.zeros((dm, dm)); o_kp = np.zeros(d); o_a = np.zeros(d); o_cc = np.zeros(d)
    fp.step_fwd(xin, h, SWa, SB, SK, Sba,
                L.Wa.data, L.ba.data, L.B.data, L.K.data, np.zeros(d), False,
                M, L.Wc.data, L.Wq.data, L.Wm.data, L.Wg1.data, L.Wg2.data,
                L.Wo.data, L.Af.data, L.Bf.data, cfg.use_memory, cfg.adapter_rank > 0,
                getattr(L.Wk, "data", L.Wk), getattr(L.Wv, "data", L.Wv),
                kp, 0.0, cfg.beta_max, cfg.mem_hetero, cfg.mem_surprise_fixed,
                cfg.mem_query_x,
                o_h, o_SWa, o_SB, o_SK, o_Sba, o_y, o_M, o_kp, o_a, o_cc, scr)
    return o_h, o_SWa, o_SB, o_SK, o_Sba, o_y, o_M, o_kp, o_a, o_cc


def main():
    nl, d, dm, V = 2, 16, 8, 64
    cfg, m = build(nl, d, dm, True, V)
    rng = np.random.default_rng(0)
    xin = rng.standard_normal(d)

    # ---------- jalur lama
    scr = np.zeros(fp.scratch_size(d, dm, cfg.expand * d, cfg.adapter_rank))
    h_all, SW_all, SB_all, SK_all, Sba_all, y_all, M_all, kp_all, a_all, cc_all = ([] for _ in range(10))
    cur = xin
    for l in range(nl):
        # state tiap lapisan mulai dari nol (PER-LAPISAN, jangan bawa sisa lapisan bawah)
        h = np.zeros(d)
        SWa = np.zeros((d, d)); SB = np.zeros((d, d)); SK = np.zeros((d, d)); Sba = np.zeros(d)
        M = np.zeros((dm, dm)); kp = np.zeros(d)
        o = ref_step_fwd_layer(m, cfg, l, cur, h, SWa, SB, SK, Sba, M, kp, scr)
        h, SWa, SB, SK, Sba, y, M, kp, a_, cc_ = o
        h_all.append(h.copy()); SW_all.append(SWa.copy()); SB_all.append(SB.copy())
        SK_all.append(SK.copy()); Sba_all.append(Sba.copy()); y_all.append(y.copy())
        M_all.append(M.copy()); kp_all.append(kp.copy()); a_all.append(a_.copy()); cc_all.append(cc_.copy())
        cur = y
    ref_logits = fp.head_fwd_lin(y @ m.P[(nl - 1, 1)].data @ np.eye(d) if False else y, m.W_out.data, m.byte_bias.data)
    # NB: top.y dipakai APA ADANYA ke kepala (bukan @P) -- lihat _chain: p1 = top.y @ P[(Ltop,1)]
    p1 = y_all[-1] @ m.P[(nl - 1, 1)].data
    ref_logits = fp.head_fwd_lin(p1, m.W_out.data, m.byte_bias.data)

    # ---------- jalur fused
    ssz = fp.state_size_per_layer(d, dm)
    st = np.zeros(nl * ssz)
    loff = np.zeros((nl, 12), dtype=np.int64)
    par = []
    names = ("Wa", "ba", "B", "K", "Wc", "Wq", "Wm", "Wg1", "Wg2", "Wo", "Af", "Bf", "Wk", "Wv")
    for l in range(nl):
        L = m.layers[l]
        row = []
        for i, nm_ in enumerate(names):
            a = getattr(L, nm_, None)
            a = getattr(a, "data", a) if a is not None else np.zeros((d, dm))
            row.append(a)
        par.append(tuple(row))
    par = tuple(par)
    bundle = (d, dm, cfg.expand * d, cfg.adapter_rank, V, nl,
              int(cfg.use_memory), int(cfg.adapter_rank > 0), int(cfg.mem_query_x),
              int(cfg.mem_hetero), int(cfg.mem_surprise_fixed), 0,
              cfg.beta_max, par, m.W_out.data, m.byte_bias.data, m.P[(nl - 1, 1)].data)
    nscr = fp.scratch_size(d, dm, cfg.expand * d, cfg.adapter_rank) + 2 * d
    scr2 = np.zeros(nscr)
    out = np.zeros(d + V + 2 * nl * d)
    fp.fused_fwd(bundle, st, loff, xin, out, scr2)

    # ---------- bandingkan
    top_base = (nl - 1) * ssz
    out_ytop = st[top_base + d + 3 * d * d + d:top_base + d + 3 * d * d + 2 * d]
    out_logits = out[d:d + V]
    errs = {}
    ytop_ref = y_all[-1]
    errs["ytop"] = float(np.abs(out_ytop - ytop_ref).max())
    errs["logits"] = float(np.abs(out_logits - ref_logits).max())
    for l in range(nl):
        b = l * ssz
        errs[f"L{l}.h"] = float(np.abs(st[b:b + d] - h_all[l]).max())
        errs[f"L{l}.y"] = float(np.abs(st[b + d + 3 * d * d + d:b + d + 3 * d * d + 2 * d] - y_all[l]).max())
        errs[f"L{l}.SWa"] = float(np.abs(st[b + d:b + d + d * d].reshape(d, d) - SW_all[l]).max())
        errs[f"L{l}.Sba"] = float(np.abs(st[b + d + 3 * d * d:b + d + 3 * d * d + d] - Sba_all[l]).max())
        errs[f"L{l}.M"] = float(np.abs(st[b + d + 3 * d * d + 2 * d:b + d + 3 * d * d + 2 * d + dm * dm].reshape(dm, dm) - M_all[l]).max())
        errs[f"L{l}.kp"] = float(np.abs(st[b + d + 3 * d * d + 2 * d + dm * dm:b + d + 3 * d * d + 3 * d + dm * dm] - kp_all[l]).max())

    worst = max(errs.values())
    for k, v in errs.items():
        flag = "  <-- BEDA" if v > 1e-12 else ""
        print(f"  {k:12s} {v:.3e}{flag}")
    print(f"\nWORST = {worst:.3e}  -> {'OK (bit-identik)' if worst < 1e-12 else 'GAGAL'}")


if __name__ == "__main__":
    main()
