"""VARIAN MIXER — apakah skema interpolasi mixer (per-orde, bobot global) kalah dari
interpolasi n-gram biasa (Witten-Bell penuh / Jelinek-Mercer)?

Mixer TMT2: logits += sum_o w_o * log p_o(ctx),  p_o = WB(orde o -> unigram)
Lantai JM  : p = JM(p_1..p_o)  (interpolasi ARITMETIK bertingkat) -> 1,925 bit (orde 5)

Kedua skema dijalankan pada tabel ONLINE yang SAMA (OnlineNgramN), jadi perbandingannya
adil dan mengisolasi SKEMA, bukan tabelnya.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.corpus import load
from tmt2.fast_learner import OnlineNgramN

LN2 = float(np.log(2))


def wb_full(ng, ctx, omax):
    """Interpolasi Witten-Bell penuh sampai orde `omax` (padanan OnlineNgramN.logprior)."""
    p = ng.uni / ng.uni.sum()
    for k in range(1, omax + 1):
        if len(ctx) < k:
            break
        c = ng.tabs[k - 1].get(ctx[-k:])
        if c is None:
            break
        d, n = c
        lam = n / (n + len(d))
        p = (1.0 - lam) * p
        for tok, cnt in d.items():
            p[tok] += lam * cnt / n
    return np.log(p)


def ce_logits(Z, Y):
    """CE (bit) dari logits Z (n,V) terhadap target Y (n,)."""
    Zc = Z - Z.max(1, keepdims=True)
    return float(np.mean(Zc.max(1) + np.log(np.exp(Zc).sum(1)) - Zc[np.arange(len(Y)), Y]) / LN2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikitext")
    ap.add_argument("--ntrain", type=int, default=200000)
    ap.add_argument("--ntest", type=int, default=6000)
    ap.add_argument("--maxorder", type=int, default=8)
    ap.add_argument("--nval", type=int, default=2000)
    a = ap.parse_args()

    raw = np.frombuffer(load(a.data), dtype=np.uint8)
    V = 256
    tr = raw[:a.ntrain]
    te = raw[a.ntrain:a.ntrain + a.ntest]

    ng = OnlineNgramN(V, a.maxorder)
    for i in range(len(tr)):
        ctx = tuple(int(x) for x in tr[max(0, i - a.maxorder):i])
        ng.update(ctx, int(tr[i]))
    print(f"data={a.data} ntrain={len(tr)} ntest={len(te)-1} tabel ONLINE orde<={a.maxorder}")

    ctxs = [tuple(int(x) for x in te[max(0, i - a.maxorder):i]) for i in range(1, len(te))]
    Y = np.asarray([int(te[i]) for i in range(1, len(te))])

    print("\n(a) WB PENUH (interpolasi aritmetik bertingkat), potong di orde omax:")
    for omax in range(1, a.maxorder + 1):
        lg = np.asarray([wb_full(ng, c, omax) for c in ctxs])
        print(f"   omax={omax}: {float(np.mean(-lg[np.arange(len(Y)), Y]) / LN2):.4f} bit", flush=True)

    print("\n(b) SKEMA MIXER: logits = sum_o w_o * log p_o(ctx)  (bobot global, cari di val):")
    P = {o: np.asarray([ng.logprior_order(c, o) for c in ctxs]) for o in range(1, a.maxorder + 1)}

    def combo(w, sl):
        return sum(w[k] * P[k + 1][sl] for k in range(a.maxorder))

    nval = min(a.nval, len(Y) // 2)
    sl = slice(0, nval)
    w = np.ones(a.maxorder)
    best_val = ce_logits(combo(w, sl), Y[sl])
    for _ in range(30):
        improved = False
        for o in range(a.maxorder):
            cur = w[o]
            for cand in np.linspace(0.0, 2.0, 41):
                w[o] = cand
                v = ce_logits(combo(w, sl), Y[sl])
                if v < best_val - 1e-6:
                    best_val, cur, improved = v, cand, True
            w[o] = cur
        if not improved:
            break
    ce_mix = ce_logits(combo(w, slice(None)), Y)
    print(f"   bobot terbaik w = {np.round(w, 3)}")
    print(f"   CE uji (skema mixer) = {ce_mix:.4f} bit")

    print("\n(c) interpolasi ARITMETIK antara WB-penuh dan skema-mixer:")
    pfull = np.exp(np.asarray([wb_full(ng, c, a.maxorder) for c in ctxs]))
    pfull /= pfull.sum(1, keepdims=True)
    Zm = combo(w, slice(None))
    pmix = np.exp(Zm - Zm.max(1, keepdims=True))
    pmix /= pmix.sum(1, keepdims=True)
    for al in (0.3, 0.5, 0.7, 0.9):
        p = al * pfull + (1 - al) * pmix
        print(f"   alpha={al}: {float(np.mean(-np.log2(p[np.arange(len(Y)), Y] + 1e-12))):.4f} bit")

    print("\n== bacaan ==")
    print(f"  lantai JM (ngram_ceiling.py, orde 5) : 1.9254 bit")
    print(f"  WB penuh (omax terbaik di atas)      : lihat (a)")
    print(f"  skema mixer (bobot global)           : {ce_mix:.4f} bit")


if __name__ == "__main__":
    main()
