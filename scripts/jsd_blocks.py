"""JSD domain per blok 1MB vs suite — uji pemutus benjolan 18M (#3, T0.6).

Untuk tiap blok 1MB raw[0:30M]: histogram bigram byte (65536-bin) ternormalisasi
+ JSD terhadap histogram suite (raw[30M:34M]). Prediksi: rerata JSD blok[12,18)
> blok[6,12) dan blok[18,30) +1 SE-blok (efek domain-urutan, keyakinan 60%).

Jalan: python scripts/jsd_blocks.py  (~2 menit, numpy murni)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.console import ensure_utf8

ensure_utf8()


def bigram_hist(a):
    a = np.ascontiguousarray(a, dtype=np.uint8).astype(np.int64)
    h = np.bincount(a[:-1] * 256 + a[1:], minlength=65536).astype(np.float64)
    return h / h.sum()


def jsd(p, q):
    m = 0.5 * (p + q)
    nz = (p > 0) & (q > 0)
    with np.errstate(divide="ignore"):
        t = np.zeros_like(p)
        t[nz] = p[nz] * np.log(p[nz] / m[nz]) + q[nz] * np.log(q[nz] / m[nz])
    return float(0.5 * t.sum() / np.log(2.0))  # bit


def main():
    with open("data/indonesian_big.bin", "rb") as f:
        raw = np.frombuffer(f.read(34_000_000), dtype=np.uint8)
    suite = raw[30_000_000:34_000_000]
    qs = bigram_hist(suite)
    rows = []
    for s in range(0, 30_000_000, 1_000_000):
        v = jsd(bigram_hist(raw[s:s + 1_000_000]), qs)
        rows.append(v)
        print(f"  blok {s//1_000_000:2d}: JSD {v:.4f} bit", flush=True)
    segs = {"[0,6)": rows[0:6], "[6,12)": rows[6:12], "[12,18)": rows[12:18],
            "[18,30)": rows[18:30]}
    for k, v in segs.items():
        print(f"  {k}: rerata {np.mean(v):.4f} (SE {np.std(v)/len(v)**0.5:.4f})",
              flush=True)


if __name__ == "__main__":
    main()
