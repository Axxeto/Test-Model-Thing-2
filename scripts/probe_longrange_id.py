"""PROBE JARAK-JAUH (Indonesia) — berapa besar retrieval sederhana menolong?

Proksi murah untuk "Kenong" (VQ-IVF) di arsitektur Gamelan: model MATCH jarak-jauh berbasis
indeks K-gram. Menjawab dua hal sebelum membangun retrieval penuh:
  1. Berapa sering konteks panjang (K byte) berulang di riwayat 200k, dan berapa akurat
     penerusnya? (plafon retrieval)
  2. Berapa bit yang benar-benar direbut bila match digabung dengan n-gram?

Protokol: riwayat = data[:200000], uji = data[200000:208000] (jendela yang SAMA dengan
`gate1_sweep.py`). Indeks dibangun atas riwayat dan diperbarui INKREMENTAL saat uji
(online, tanpa melihat masa depan).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.corpus import load

LN2 = float(np.log(2))


def ngram_logp(tr, te, order=5, lam=0.5, V=256):
    """log p (nat) per posisi uji untuk n-gram JM, tabel dari tr (beku)."""
    tabs, tots = [], []
    for o in range(1, order + 1):
        tab, tot = {}, {}
        for i in range(o, len(tr)):
            c = tuple(int(x) for x in tr[i - o:i])
            b = int(tr[i])
            tab.setdefault(c, np.zeros(V))[b] += 1.0
            tot[c] = tot.get(c, 0.0) + 1.0
        tabs.append(tab)
        tots.append(tot)
    uni = np.bincount(np.asarray(tr), minlength=V).astype(float)
    uni = (uni + 1e-9) / (uni.sum() + 1e-9 * V)
    out = np.zeros(len(te))
    for i in range(len(te)):
        ctx = tuple(int(x) for x in te[max(0, i - order):i])
        p = uni
        for o in range(1, order + 1):
            c = ctx[-o:]
            if len(c) < o:
                continue
            cc = tabs[o - 1].get(c)
            if cc is None:
                continue
            p = lam * (cc / tots[o - 1][c]) + (1 - lam) * p
        out[i] = np.log(p[int(te[i])] + 1e-12)
    return out


def match_predict(tr, te, K=16, maxcand=8, maxtab=64):
    """Match jarak-jauh: indeks K-gram -> posisi; prediksi = penerus kemunculan termirip.

    Mengembalikan (prediksi per posisi atau -1, panjang match per posisi, akurasi).
    Indeks diperbarui inkremental (online).
    """
    hist = bytearray(int(x) for x in tr)
    index = {}
    for p in range(0, len(hist) - K):
        key = bytes(hist[p:p + K])
        lst = index.get(key)
        if lst is None:
            index[key] = [p]
        else:
            lst.append(p)
            if len(lst) > maxcand:
                lst.pop(0)
    preds = np.full(len(te), -1, dtype=np.int64)
    lens = np.zeros(len(te), dtype=np.int64)
    for i in range(len(te)):
        n = len(hist)
        if n >= K:
            key = bytes(hist[n - K:n])
            lst = index.get(key)
            if lst:
                best_len, best_pred = 0, -1
                for p in reversed(lst):
                    if p + K >= n:              # penerus belum teramati -> tolak
                        continue
                    # panjang match TOTAL = K + (perluasan ke BELAKANG dari jendela K-gram).
                    # Perbandingan harus sejajar: hist[p-1-L] vs hist[n-K-1-L] (BUKAN hist[n-1-L]).
                    L = 0
                    while L < maxtab and p - 1 - L >= 0 and n - K - 1 - L >= 0 \
                            and hist[p - 1 - L] == hist[n - K - 1 - L]:
                        L += 1
                    if L > best_len:
                        best_len, best_pred = L, int(hist[p + K])
                if best_pred >= 0:
                    preds[i] = best_pred
                    lens[i] = K + best_len
        b = int(te[i])
        hist.append(b)
        if len(hist) >= K:
            key = bytes(hist[len(hist) - K:])
            lst = index.get(key)
            if lst is None:
                index[key] = [len(hist) - K]
            else:
                lst.append(len(hist) - K)
                if len(lst) > maxcand:
                    lst.pop(0)
    return preds, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/indonesian_1m_pure.txt")
    ap.add_argument("--ntrain", type=int, default=200000)
    ap.add_argument("--ntest", type=int, default=8000)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--order", type=int, default=5)
    a = ap.parse_args()

    raw = np.frombuffer(load(a.data), dtype=np.uint8)
    tr = raw[:a.ntrain]
    te = raw[a.ntrain:a.ntrain + a.ntest]
    Y = np.asarray([int(x) for x in te])
    print(f"data={a.data} ntrain={len(tr)} ntest={len(te)} K={a.K} n-gram orde={a.order}")

    lp = ngram_logp(tr, te, order=a.order)
    ce_ng = float(-lp.mean() / LN2)
    print(f"n-gram JM-{a.order} (beku)        : {ce_ng:.4f} bit")

    preds, lens = match_predict(tr, te, K=a.K)
    have = preds >= 0
    print(f"\nmatch ditemukan                  : {have.mean()*100:.1f}% posisi")
    if have.any():
        acc = float((preds[have] == Y[have]).mean())
        print(f"akurasi penerus bila match ada   : {acc*100:.1f}%  (lantai tebak-acak "
              f"{100/256:.1f}%)")
        print(f"panjang match (median/maks)      : {int(np.median(lens[have]))} / {int(lens.max())}")

    print(f"\n{'gabungan (bobot match)':<30}{'CE (bit)':>10}")
    print("-" * 40)
    for w in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0):
        lp2 = lp.copy()
        m = have & (w > 0)
        if m.any():
            # campuran: log((1-w)*p_ng + w*delta(pred))
            png = np.exp(lp[m])
            lp2[m] = np.log(np.maximum((1 - w) * png + w * (preds[m] == Y[m]), 1e-12))
        ce = float(-lp2.mean() / LN2)
        print(f"{'w=' + str(w):<30}{ce:>10.4f}")

    print("\nakurasi per PANJANG match (inti pertanyaan: apakah match panjang akurat?):")
    print(f"  {'rentang L':<14}{'n':>7}{'% posisi':>10}{'akurasi':>10}")
    for lo, hi in ((a.K, 15), (16, 31), (32, 63), (64, 10 ** 9)):
        m = have & (lens >= lo) & (lens <= hi)
        if m.sum() == 0:
            continue
        lab = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        print(f"  {lab:<14}{int(m.sum()):>7}{m.mean()*100:>9.1f}%"
              f"{(preds[m] == Y[m]).mean()*100:>9.1f}%")

    print("\n== pembanding ==")
    print("  TMT2+mixer (3 seed)            : 2.4809")
    print("  bz2 -9 kondisional (target)    : 2.3700")


if __name__ == "__main__":
    main()
