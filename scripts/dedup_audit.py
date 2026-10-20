"""Audit duplikat latih-vs-suite H1 (tanpa kode produksi, ~5-10 mnt).

Hash shingle 32B stride-16 dari 30M latih -> set; fraksi duplikat per jendela
suite 20x200kB + korelasi gain-kapasitas vs duplikat. Bila gain pada jendela
non-duplikat >=70% gain total -> H1 gugur (gain itu nyata).

Jalan: python scripts/dedup_audit.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.console import ensure_utf8

ensure_utf8()


def shingles(a, w=32, stride=16):
    a = np.ascontiguousarray(a, dtype=np.uint8)
    n = (len(a) - w) // stride + 1
    # FNV-1a 64-bit per shingle (loop Python dioptimasi via stride besar? tidak:
    # pakai rolling sederhana di numpy: view 2D lalu hash kolom)
    wins = np.lib.stride_tricks.as_strided(
        a, shape=(n, w), strides=(a.strides[0] * stride, a.strides[0]))
    h = np.zeros(n, dtype=np.uint64) + np.uint64(14695981039346656037)
    for j in range(w):
        h = (h ^ wins[:, j].astype(np.uint64)) * np.uint64(1099511628211)
    return set(h.tolist())


def main():
    t0 = time.perf_counter()
    with open("data/indonesian_big.bin", "rb") as f:
        raw = np.frombuffer(f.read(34_000_000), dtype=np.uint8)
    tr, suite = raw[:30_000_000], raw[30_000_000:34_000_000]
    print("hash latih...", flush=True)
    S = shingles(tr)
    print(f"  {len(S)} shingle unik ({time.perf_counter()-t0:.0f}s)", flush=True)
    fracs = []
    for i in range(20):
        w = suite[i * 200_000:(i + 1) * 200_000]
        n = (len(w) - 32) // 16 + 1
        wins = np.lib.stride_tricks.as_strided(
            np.ascontiguousarray(w, dtype=np.uint8), shape=(n, 32),
            strides=(1 * 16, 1))
        h = np.zeros(n, dtype=np.uint64) + np.uint64(14695981039346656037)
        for j in range(32):
            h = (h ^ wins[:, j].astype(np.uint64)) * np.uint64(1099511628211)
        hit = sum(1 for x in h.tolist() if x in S)
        fracs.append(hit / n)
        print(f"  jendela {i:2d}: dup {hit/n:.3f}", flush=True)
    fracs = np.array(fracs)
    print(f"rerata dup: {fracs.mean():.3f} (min {fracs.min():.3f}, max {fracs.max():.3f})",
          flush=True)
    print("PUTUSAN: jendela non-duplikat (<10%) = "
          f"{int((fracs < 0.10).sum())}/20", flush=True)


if __name__ == "__main__":
    main()
