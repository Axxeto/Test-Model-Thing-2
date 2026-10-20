"""LANTAI KNESER-NEY — apakah smoothing KN mengalahkan WB/JM pada byte WikiText?

Mixer memakai Witten-Bell per-orde. Bila KN (absolute discounting + continuation unigram)
jauh lebih baik, maka mengganti smoothing mixer adalah perbaikan kualitas konkret.

Tabel dari korpus LATIH, diuji out-of-sample pada bagian tahan.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.corpus import load


def build(train, K):
    tabs = [None] * (K + 1)
    for k in range(1, K + 1):
        tab = {}
        for i in range(k, len(train)):
            c = tuple(int(x) for x in train[i - k:i])
            w = int(train[i])
            d = tab.get(c)
            if d is None:
                d = tab[c] = {}
            d[w] = d.get(w, 0.0) + 1.0
        tabs[k] = tab
    return tabs


def cont_unigram(train, V):
    """N1+(.w): jumlah konteks (byte sebelum) yang berbeda yang mendahului w."""
    cont = np.zeros(V)
    seen = set()
    for i in range(1, len(train)):
        u, w = int(train[i - 1]), int(train[i])
        key = (u, w)
        if key not in seen:
            seen.add(key)
            cont[w] += 1.0
    return cont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikitext")
    ap.add_argument("--ntrain", type=int, default=200000)
    ap.add_argument("--ntest", type=int, default=6000)
    ap.add_argument("--maxorder", type=int, default=8)
    ap.add_argument("--Ds", default="0.5,0.75,1.0")
    a = ap.parse_args()

    raw = np.frombuffer(load(a.data), dtype=np.uint8)
    V = 256
    tr = raw[:a.ntrain]
    te = raw[a.ntrain:a.ntrain + a.ntest]
    print(f"data={a.data} ntrain={len(tr)} ntest={len(te)-1}")
    tabs = build(tr, a.maxorder)
    cont = cont_unigram(tr, V)
    unorm = cont.sum()

    def rec(order, h, w, D):
        order = min(order, len(h))
        if order == 0:
            return cont[w] / unorm
        d = tabs[order].get(h)
        lower = rec(order - 1, h[1:], w, D)
        if not d:
            return lower
        ch = sum(d.values())
        n1 = len(d)
        return max(d.get(w, 0.0) - D, 0.0) / ch + (D * n1 / ch) * lower

    for D in [float(x) for x in a.Ds.split(",")]:
        row = []
        for order in range(1, a.maxorder + 1):
            ce = 0.0
            for i in range(1, len(te)):
                h = tuple(int(x) for x in te[max(0, i - order):i])
                w = int(te[i])
                ce += -np.log2(rec(order, h, w, D) + 1e-12)
            row.append(ce / (len(te) - 1))
        print(f"D={D}: " + " ".join(f"o{o}={c:.4f}" for o, c in zip(range(1, a.maxorder + 1), row)),
              flush=True)

    print("\n== pembanding ==")
    print("  JM terbaik (ngram_ceiling.py)      : 1.9254 bit (orde 5)")
    print("  WB-penuh terbaik (mixer_variants)  : 1.9259 bit (omax 5)")
    print("  mixer+model ONLINE (200k)          : 1.8440 bit")


if __name__ == "__main__":
    main()
