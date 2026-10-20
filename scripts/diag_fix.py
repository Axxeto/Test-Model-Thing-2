"""Uji: apa yang MENOLONG degradasi BPB-uji pada korpus besar?

Diagnostik `diag_overwrite.py` menunjukkan kurva U: BPB-uji turun sampai tabel jenuh lalu
NAIK monoton, dan semua `table_bits` konvergen ke ~2,28 @30 jt byte. Analisis aliran:
  * nctx=11 konteks x 8 bit = 88 lookup/byte.
  * tb=25: 30 jt byte -> ~79 tulis per slot. Dengan COUNT_CAP=1023 (~2^10) tiap penghitung
    di-reset-separuh ~7x per slot -> statistik konteks lama terus tergerus konteks baru.
Kandidat penyebab & solusinya:
  1. `COUNT_CAP` terlalu kecil  -> uji naikkan (1023 -> 4095/16383).
  2. tumbukan checksum 8-bit    -> uji fungsi hash berbeda (kubik) tetap 8-bit checksum.
  3. tabel kurang besar          -> uji tb=28 (2,1 GB) [mahal, opsional].
  4. peluruhan lr mixer          -> uji lr dinaikkan/diturunkan (kontrol negatif).

Protokol: prefix latih 30 jt byte, jendela uji 20 rb SETELAH prefix (tak beririsan — dijaga
assert). Ukur di 3 titik: 10 / 20 / 30 jt byte.

Jalan:
  python scripts/diag_fix.py --ntrain 30000000 --table_bits 25 --count_cap 1023,4095,16383
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

import tmt2.stat_core as sc
from tmt2.stat_core import StatCore

CORPUS = "data/indonesian_big.bin"


def run_one(raw, ntrain, ntest, tb, cap, orders, lr, probes):
    """Latih satu konfigurasi, ukur BPB-uji di titik-titik probe. Mengembalikan list dict.

    CATATAN PENTING (bug yang dihindari): semula ``COUNT_CAP`` adalah konstanta modul yang
    dibaca numba SSAT COMPILE, jadi menetapkannya di runtime tidak berpengaruh (cap 1023 vs
    4095 -> BPB identik). Sekarang ``count_cap`` adalah PARAMETER `stat_step`/`StatCore`,
    jadi nilainya benar-benar dipakai. Jangan kembali ke konstanta global untuk knob ini.
    """
    tr = raw[:ntrain]
    te = raw[ntrain:ntrain + ntest]
    assert ntrain + ntest <= raw.size
    m = StatCore(orders=orders, table_bits=tb, lr=lr, use_apm=True,
                 wsel_bits=12, sparse=False, word=True, count_cap=cap)
    out = []
    n = 0
    t0 = time.perf_counter()
    next_probe = probes[0]
    pi = 0
    for b in tr:
        m.step(b)
        n += 1
        if pi < len(probes) and n >= next_probe:
            ce = float(m.eval_stream(te).mean())
            fill = np.count_nonzero(m.tab) / m.tab.size
            out.append(dict(table_bits=tb, count_cap=int(cap), n=n, bpb=ce, table_fill=fill,
                            bps=n / max(time.perf_counter() - t0, 1e-9)))
            pi += 1
            next_probe = probes[pi] if pi < len(probes) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--ntrain", type=int, default=30000000)
    ap.add_argument("--ntest", type=int, default=20000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--count_cap", default="1023,4095,16383")
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--jsonl", default="results/diag_fix.jsonl")
    a = ap.parse_args()

    caps = [int(x) for x in a.count_cap.split(",")]
    orders = tuple(int(x) for x in a.orders.split(","))
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    probes = [a.ntrain // 3, 2 * a.ntrain // 3, a.ntrain]
    print(f"latih {a.ntrain} B | uji(setelah prefix) {a.ntest} B | tb={a.table_bits} "
          f"| probe di {probes}", flush=True)
    print(f"COUNT_CAP semula = {sc.COUNT_CAP}", flush=True)
    os.makedirs("results", exist_ok=True)

    rows = []
    print(f"\n{'cap':>7} {'10jt':>9} {'20jt':>9} {'30jt':>9}  (BPB-uji)", flush=True)
    for cap in caps:
        rs = run_one(raw, a.ntrain, a.ntest, a.table_bits, cap, orders, a.lr, probes)
        rows.extend(rs)
        vals = {r["n"]: r["bpb"] for r in rs}
        print(f"{cap:>7} {vals.get(probes[0], float('nan')):>9.4f} "
              f"{vals.get(probes[1], float('nan')):>9.4f} "
              f"{vals.get(probes[2], float('nan')):>9.4f}", flush=True)

    with open(a.jsonl, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
