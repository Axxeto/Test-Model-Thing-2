"""LANTAI N-GRAM — berapa CE terbaik yang bisa dicapai model konteks-lokal pada byte?

Mixer TMT2 (Pilar 2) mengklaim jadi pekerja kualitas. Berapa plafonnya? Ukur n-gram
byte terinterpolasi (Jelinek-Mercer) orde 1..8 pada korpus & split yang sama, lalu
bandingkan dengan (a) mixer-murni dan (b) model penuh.

Tabel dari korpus LATIH, diuji out-of-sample pada bagian tahan. Lamda JM disapu.

Penting: bila mixer << lantai n-gram, ada bit yang bisa direbut hanya dengan
memperbaiki mixer (tanpa menyentuh SSM).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.corpus import load


def build_tables(train, order, V):
    tabs, tots = [], []
    for o in range(1, order + 1):
        tab, tot = {}, {}
        for i in range(o, len(train)):
            c = tuple(int(x) for x in train[i - o:i])
            b = int(train[i])
            tab.setdefault(c, np.zeros(V))[b] += 1.0
            tot[c] = tot.get(c, 0.0) + 1.0
        tabs.append(tab)
        tots.append(tot)
    return tabs, tots


def ce_jm(uni, tabs, tots, ctxs, curs, order, lam):
    ce = 0.0
    for ctx, b in zip(ctxs, curs):
        ctx = tuple(int(x) for x in ctx)
        p = uni
        for o in range(1, order + 1):
            c = ctx[-o:]
            if len(c) < o:
                continue
            cc = tabs[o - 1].get(c)
            if cc is None:
                continue
            p = lam * (cc / tots[o - 1][c]) + (1 - lam) * p
        ce += -np.log2(p[b] + 1e-12)
    return ce / len(curs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikitext")
    ap.add_argument("--ntrain", type=int, default=200000)
    ap.add_argument("--ntest", type=int, default=8000)
    ap.add_argument("--maxorder", type=int, default=8)
    ap.add_argument("--lams", default="0.3,0.5,0.7,0.9")
    a = ap.parse_args()

    raw = np.frombuffer(load(a.data), dtype=np.uint8)
    V = 256
    tr = raw[:a.ntrain]
    te = raw[a.ntrain:a.ntrain + a.ntest]
    ctxs = [te[max(0, i - a.maxorder):i] for i in range(1, len(te))]
    curs = [int(te[i]) for i in range(1, len(te))]
    print(f"data={a.data} ntrain={len(tr)} ntest={len(te)-1} V={V}")
    uni = np.bincount(tr, minlength=V).astype(float)
    uni = (uni + 1e-9) / (uni.sum() + 1e-9 * V)

    lams = [float(x) for x in a.lams.split(",")]
    print(f"\n{'order':>5}" + "".join(f"{('lam=%.1f' % l):>10}" for l in lams) + f"{'terbaik':>10}")
    print("-" * (5 + 10 * (len(lams) + 1)))
    best = {}
    for order in range(1, a.maxorder + 1):
        t0 = time.time()
        tabs, tots = build_tables(tr, order, V)
        row, bv = [], 9.9
        for l in lams:
            c = ce_jm(uni, tabs, tots, ctxs, curs, order, l)
            row.append(c)
            bv = min(bv, c)
        best[order] = bv
        print(f"{order:>5}" + "".join(f"{c:>10.4f}" for c in row) + f"{bv:>10.4f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)

    print("\n== bacaan ==")
    b = min(best, key=best.get)
    print(f"  lantai n-gram byte terbaik: orde {b} = {best[b]:.4f} bit")
    print(f"  (pembanding) model SSM saja di d=32/L=2, ntrain 200k: ~2.97 bit")
    print(f"  (pembanding) BPE unigram adil (dari STATUS): 2.115 bit")


if __name__ == "__main__":
    main()
