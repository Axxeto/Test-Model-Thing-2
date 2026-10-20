"""Uji eksak: fused_bwd vs jalur lama (head_bwd_lin_acc + n_layers x flush_rec), SATU langkah.

Protocol: satu langkah maju dari state nol -> fused_fwd mengisi arena state -> backward
lewat kedua jalur dengan glo yang sama -> bandingkan arena gradien G bit-demi-bit.
JANGAN uji lintasan panjang (chaos online).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2 import fastpath as fp

NAMES = ("Wa", "ba", "B", "K", "Wc", "Wq", "Wm", "Wg1", "Wg2", "Wo", "Af", "Bf", "Wk", "Wv")


def build_bundle(cfg, m, nl, d, dm, V, usedK=0, deep_gate=1, oWout=0, oBias=0, oP=0):
    ed = cfg.expand * d
    r = cfg.adapter_rank
    par = []
    for l in range(nl):
        L = m.layers[l]
        par.append(tuple(getattr(getattr(L, nm_, None), "data", np.zeros((d, dm))) for nm_ in NAMES))
    return (d, dm, ed, r, V, nl, int(cfg.use_memory), int(r > 0), int(cfg.mem_query_x),
            int(cfg.mem_hetero), int(cfg.mem_surprise_fixed), 0, cfg.beta_max,
            tuple(par), m.W_out.data, m.byte_bias.data, m.P[(nl - 1, 1)].data,
            usedK, deep_gate, oWout, oBias, oP)


def slices(st, l, ssz, d, dm):
    base = l * ssz
    return dict(
        h=st[base:base + d],
        SWa=st[base + d:base + d + d * d].reshape(d, d),
        SB=st[base + d + d * d:base + d + 2 * d * d].reshape(d, d),
        SK=st[base + d + 2 * d * d:base + d + 3 * d * d].reshape(d, d),
        Sba=st[base + d + 3 * d * d:base + d + 3 * d * d + d],
        y=st[base + d + 3 * d * d + d:base + d + 3 * d * d + 2 * d],
        M=st[base + d + 3 * d * d + 2 * d:base + d + 3 * d * d + 2 * d + dm * dm].reshape(dm, dm),
        a=st[base + d + 3 * d * d + 3 * d + dm * dm:base + d + 3 * d * d + 4 * d + dm * dm],
        cc=st[base + d + 3 * d * d + 4 * d + dm * dm:base + d + 3 * d * d + 5 * d + dm * dm],
    )


def main():
    nl, d, dm, V = 2, 16, 8, 64
    cfg = Config.best(d=d, n_layers=nl, vocab=V, d_mem=dm, use_memory=True,
                      head="linear", deep_grad=True)
    m = TMT2(cfg)
    ed, r = cfg.expand * d, cfg.adapter_rank
    ssz = fp.state_size_per_layer(d, dm)
    st = np.zeros(nl * ssz)
    ntot = 100_000
    loff = np.zeros((nl, 12), dtype=np.int64)
    for l in range(nl):
        for i in range(12):
            loff[l, i] = (l * 12 + i) * 400
    oWout, oBias, oP = 96000, 98000, 30000

    scr_f = np.zeros(fp.scratch_size(d, dm, ed, r) + 2 * d)
    rng = np.random.default_rng(1)
    xin = rng.standard_normal(d)
    out = np.zeros(d + V + 2 * nl * d)
    bundle = build_bundle(cfg, m, nl, d, dm, V, 0, 1, oWout, oBias, oP)
    fp.fused_fwd(bundle, st, loff, xin, out, scr_f)
    logits = out[d:d + V]
    b = 7
    glo = fp.softmax_grad_target(logits, b, 1.0)

    # ---------- fused_bwd
    Gf = np.zeros(ntot)
    gEf = np.zeros((V, d))
    scrb = np.zeros(fp.scratch_size_bwd(d, dm, ed, r) + 8 * d + V + 3 * d)
    fp.fused_bwd(bundle, st, loff, xin, glo, b, -1, Gf, gEf, scrb)

    # ---------- jalur lama
    Gref = np.zeros(ntot)
    gEref = np.zeros((V, d))
    Ltop = nl - 1
    s = slices(st, Ltop, ssz, d, dm)
    Ptop = m.P[(Ltop, 1)].data
    gy, gP_ref, gv = fp.head_bwd_lin_acc(s["y"] @ Ptop, m.W_out.data, glo, 1.0, Ptop, s["y"], Gref, oWout)
    Gref[oBias:oBias + V] += glo  # dCE/d(byte_bias) = w*glo
    # gP sesungguhnya: dari head_bwd_lin_acc, gP = outer(y, g_p) -> kita punya gv=glo,
    # tapi g_p tidak dikembalikan. Hitung ulang g_p dari logika head_bwd_lin_acc.
    # -> Lebih aman: pakai referensi P grad via jalur yang sama seperti fused (outer(y,gp)).
    # Ambil gp dengan replay l2 norm.
    p_full = s["y"] @ Ptop
    nrm = np.sqrt((p_full * p_full).sum() + 1e-8)
    pv = p_full / nrm
    g_s = m.W_out.data.T @ glo
    dg = g_s @ pv
    gp = (g_s - pv * dg) / nrm
    Gref[oP:oP + d * d] += np.outer(s["y"], gp).reshape(-1)

    scrr = np.zeros(fp.scratch_size_bwd(d, dm, ed, r))
    gy_cur = gy.copy()
    for l in range(nl - 1, -1, -1):
        sl = slices(st, l, ssz, d, dm)
        x = xin if l == 0 else slices(st, l - 1, ssz, d, dm)["y"]
        L = m.layers[l]
        lo = loff[l]
        o_gam = np.zeros(dm); o_q = np.zeros(dm); o_gx = np.zeros(d)
        gAf = np.zeros((d, r)); gBf = np.zeros((r, d))
        fp.flush_rec(sl["h"], x, sl["M"], gy_cur, L.Wc.data, L.Wq.data, L.Wm.data,
                     L.Wg1.data, L.Wg2.data, L.Wo.data, L.Af.data, L.Bf.data,
                     sl["SWa"], sl["SB"], sl["SK"], sl["Sba"], L.Wa.data, L.B.data,
                     sl["a"], sl["cc"], False, cfg.use_memory, r > 0, cfg.mem_query_x, 1,
                     Gref, gAf, gBf,
                     lo[4], lo[5], lo[6], lo[7], lo[8], lo[9], lo[0], lo[2], lo[1], lo[3],
                     -1, o_gam, o_q, o_gx, scrr)
        if l > 0:
            gy_cur = o_gx + gy_cur

    # ---------- bandingkan
    errs = {"G_all": float(np.abs(Gf - Gref).max())}
    for l in range(nl):
        lo = loff[l]
        for ri, nm_ in enumerate(NAMES[:10]):
            o = lo[ri]
            errs[f"L{l}.{nm_}"] = float(np.abs(Gf[o:o + 400] - Gref[o:o + 400]).max())
    errs["W_out"] = float(np.abs(Gf[oWout:oWout + V * d] - Gref[oWout:oWout + V * d]).max())
    errs["byte_bias"] = float(np.abs(Gf[oBias:oBias + V] - Gref[oBias:oBias + V]).max())
    errs["P_top"] = float(np.abs(Gf[oP:oP + d * d] - Gref[oP:oP + d * d]).max())
    worst = max(errs.values())
    for k, v in errs.items():
        print(f"  {k:12s} {v:.3e}{'  <-- BEDA' if v > 1e-12 else ''}")
    print(f"\nWORST = {worst:.3e} -> {'OK (bit-identik)' if worst < 1e-12 else 'GAGAL'}")


if __name__ == "__main__":
    main()
