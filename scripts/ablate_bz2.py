"""ABLASI PIPELINE bz2 — dari mana keunggulannya di korpus Indonesia?

bz2 = RLE1 -> BWT -> MTF -> RLE2 -> Huffman(+selectors). Yang mau dipisahkan:
  (a) BWT (Burrows-Wheeler: MENGURUTKAN konteks -> pemodelan konteks tak-terbatas)
  (b) MTF + entropy coding (pemodelan statistik lokal)

Diagnostik inti: entropi ORDER-0 dari data SETELAH tiap tahap.

⚠️ DUA JEBAKAN yang sudah dikoreksi di sini:
  1. **H0(BWT(x)) == H0(x) SELALU** — BWT hanya permutasi, jadi entropi order-0-nya invarian.
     Yang bermakna adalah H0 setelah MTF: `H0(MTF(BWT(x)))`. (Versi pertama skrip ini
     melaporkan H0(BWT) dan angkanya persis sama dengan H0 mentah — itu bukan temuan.)
  2. **H_k in-sample SANGAT optimistis.** Dihitung pada data yang sama yang dipakai
     mengestimasi; pada order-5 nilainya 0,92 sedangkan out-of-sample (ngram_ceiling.py)
     2,54. Kolom itu hanya indikator relatif — JANGAN dibandingkan dengan angka model.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.corpus import load

LN2 = float(np.log(2))


def h0(seq):
    c = np.bincount(np.asarray(seq, dtype=np.int64), minlength=256).astype(float)
    c = c[c > 0]
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def hk(seq, k):
    """Entropi order-k (byte): H(b_t | b_{t-k..t-1}) empiris, dihitung dari pasangan unik."""
    a = np.asarray(seq, dtype=np.int64)
    if k == 0:
        return h0(a)
    n = len(a) - k
    code = np.zeros(n, dtype=np.int64)
    for j in range(k):
        code = code * 256 + a[j:j + n]
    nxt = a[k:]
    uniq_ctx, inv_ctx, cnt_ctx = np.unique(code, return_inverse=True, return_counts=True)
    arr = np.stack([code, nxt], axis=1)
    uniq2, cnt2 = np.unique(arr, axis=0, return_counts=True)
    ci = np.searchsorted(uniq_ctx, uniq2[:, 0])
    H = -np.sum((cnt2 / n) * np.log2(cnt2 / cnt_ctx[ci]))
    return float(H)


def suffix_array(s):
    """Suffix array via prefix doubling (numpy). O(n log^2 n)."""
    n = len(s)
    s = np.asarray(s, dtype=np.int64)
    rank = s.copy()
    sa = np.arange(n)
    k = 1
    while True:
        key2 = np.where(sa + k < n, rank[np.minimum(sa + k, n - 1)], -1)
        order = np.lexsort((key2, rank[sa]))
        sa = sa[order]
        prev, cur = sa[:-1], sa[1:]
        p2 = np.where(prev + k < n, rank[np.minimum(prev + k, n - 1)], -1)
        c2 = np.where(cur + k < n, rank[np.minimum(cur + k, n - 1)], -1)
        diff = (rank[prev] != rank[cur]) | (p2 != c2)
        newrank = np.empty(n, dtype=np.int64)
        newrank[sa[0]] = 0
        newrank[sa[1:]] = np.cumsum(diff)
        rank = newrank
        if rank[sa[-1]] == n - 1:
            break
        k *= 2
        if k > n:
            break
    return sa


def bwt(s):
    s = np.asarray(s, dtype=np.int64)
    sa = suffix_array(s)
    return s[(sa - 1) % len(s)]


def mtf(s):
    s = np.asarray(s, dtype=np.int64)
    out = np.empty(len(s), dtype=np.int64)
    order = list(range(256))
    pos = list(range(256))
    for i, b in enumerate(s):
        p = pos[b]
        out[i] = p
        if p > 0:
            order.pop(p)
            order.insert(0, b)
            for j in range(p + 1):
                pos[order[j]] = j
    return out


def rle1(s):
    """RLE1 gaya bz2: b -> (b, run-1), run 1..255."""
    s = np.asarray(s, dtype=np.int64)
    out = []
    i = 0
    n = len(s)
    while i < n:
        b = s[i]
        run = 1
        while i + run < n and s[i + run] == b and run < 255:
            run += 1
        out.append(b)
        out.append(run - 1)
        i += run
    return np.asarray(out, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/indonesian_1m_pure.txt")
    ap.add_argument("--n", type=int, default=208000)
    ap.add_argument("--orders", default="1,2,3,4,5,8")
    a = ap.parse_args()

    raw = np.frombuffer(load(a.data), dtype=np.uint8)[:a.n]
    print(f"korpus={a.data} n={len(raw)}")
    print(f"\n{'tahap':<34}{'H order-0 (bit/byte)':>22}")
    print("-" * 56)
    print(f"{'mentah':<34}{h0(raw):>22.4f}")
    print(f"{'  (H_k di bawah = IN-SAMPLE, optimistis)':<34}{'':>22}")
    for k in [int(x) for x in a.orders.split(",")]:
        t0 = time.time()
        print(f"{'mentah, entropi order-' + str(k):<34}{hk(raw, k):>22.4f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    b = bwt(raw)
    print(f"\n{'BWT(mentah) [= permutasi]':<34}{h0(b):>22.4f}   (transform {time.time()-t0:.0f}s)")
    print(f"{'  ^ invarian: H0(BWT)==H0(mentah)':<34}{'':>22}")
    t0 = time.time()
    print(f"{'MTF(BWT(mentah))  <-- BERMAKNA':<34}{h0(mtf(b)):>22.4f}   ({time.time()-t0:.0f}s)",
          flush=True)

    r = rle1(raw)
    print(f"\n{'RLE1 panjang':<34}{len(r):>22d}   (dari {len(raw)})")
    t0 = time.time()
    br = bwt(r)
    print(f"{'BWT(RLE1(mentah))':<34}{h0(br):>22.4f}   (transform {time.time()-t0:.0f}s)",
          flush=True)
    t0 = time.time()
    print(f"{'MTF(BWT(RLE1(mentah)))':<34}{h0(mtf(br)):>22.4f}   ({time.time()-t0:.0f}s)",
          flush=True)

    print("\n== pembanding (semua OUT-OF-SAMPLE kecuali disebut lain) ==")
    print("  H0 mentah                      : 4.4024")
    print("  n-gram JM orde-3 (beku)        : 2.9129")
    print("  n-gram JM orde-5 (beku)        : 2.5400")
    print("  n-gram JM orde-8 (beku)        : 2.6873")
    print("  H0(MTF(BWT)) [in-sample, 256 par] : 2.5847")
    print("  TMT2+mixer (3 seed)            : 2.4809")
    print("  bz2 -9 KONDISIONAL (target)    : 2.3700")


if __name__ == "__main__":
    main()
